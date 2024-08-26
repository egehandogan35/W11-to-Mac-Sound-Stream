use serde::Serialize;
use std::error::Error;
use std::ffi::OsString;
use std::io::Write;
use std::net::TcpStream;
use std::os::windows::ffi::OsStringExt;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::time::Duration;
use std::{ptr, thread};
use winapi::shared::guiddef::GUID;
use winapi::shared::mmreg::{
    WAVEFORMATEX, WAVEFORMATEXTENSIBLE, WAVE_FORMAT_EXTENSIBLE, WAVE_FORMAT_PCM,
};
use winapi::shared::winerror::{FAILED, S_FALSE, S_OK};
use winapi::um::audioclient::{
    IAudioCaptureClient, IAudioClient, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_S_BUFFER_EMPTY,
};
use winapi::um::audiosessiontypes::{AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK};
use winapi::um::combaseapi::{
     CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINITBASE_MULTITHREADED,
};
use winapi::um::mmdeviceapi::{
    eCapture, ERole, IMMDevice,
    IMMDeviceEnumerator,
};
use winapi::um::propidl::PROPVARIANT;
use winapi::um::strmif::REFERENCE_TIME;
use winapi::{Class, Interface};

use crossbeam::queue::SegQueue;

#[derive(Serialize, Debug)]
struct DeviceDetails {
    sample_rate: u32,
    bit_depth: u16,
    channels: u16,
}

struct CircularBuffer {
    buffer: Vec<u8>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
}

impl CircularBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
        }
    }

    fn write(&mut self, samples: &[u8]) {
        for &sample in samples {
            self.buffer[self.write_pos % self.capacity] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    fn ready_to_send(&self) -> bool {
        let unread_samples = if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.write_pos + self.capacity - self.read_pos
        };
        unread_samples >= self.capacity / 2
    }

    fn read_to_vec(&mut self, vec: &mut Vec<u8>) {
        while self.read_pos != self.write_pos {
            vec.push(self.buffer[self.read_pos % self.capacity]);
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    fn reset(&mut self) {
        println!("Resetting circular buffer.");
        self.read_pos = 0;
        self.write_pos = 0;
    }
}

fn create_wave_format(sample_rate: u32, bit_depth: u16, channels: u16) -> *mut WAVEFORMATEX {
    let format = WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_PCM,
        nChannels: channels,
        nSamplesPerSec: sample_rate,
        nAvgBytesPerSec: (sample_rate * channels as u32 * bit_depth as u32 / 8),
        nBlockAlign: (channels * bit_depth / 8) as u16,
        wBitsPerSample: bit_depth,
        cbSize: 0,
    };
    Box::into_raw(Box::new(format))
}

fn activate_audio_client(device: *mut IMMDevice) -> Result<*mut IAudioClient, Box<dyn Error>> {
    let mut audio_client: *mut IAudioClient = std::ptr::null_mut();
    let hr = unsafe {
        (*device).Activate(
            &IAudioClient::uuidof(),
            CLSCTX_ALL,
            std::ptr::null_mut(),
            &mut audio_client as *mut _ as *mut _,
        )
    };
    if FAILED(hr) {
        Err(format!("Failed to activate audio client. HRESULT: 0x{:x}", hr).into())
    } else {
        Ok(audio_client)
    }
}

fn get_mix_format(audio_client: *mut IAudioClient) -> Result<*mut WAVEFORMATEX, Box<dyn Error>> {
    let mut mix_format: *mut WAVEFORMATEX = std::ptr::null_mut();
    let hr_format = unsafe { (*audio_client).GetMixFormat(&mut mix_format) };
    if FAILED(hr_format) {
        Err(format!("Failed to get mix format. HRESULT: 0x{:x}", hr_format).into())
    } else {
        println!("Mix Format:");
        print_audio_format_details(mix_format);
        Ok(mix_format)
    }
}

fn initialize_audio_client(
    audio_client: *mut IAudioClient,
    mix_format: *mut WAVEFORMATEX,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut closest_match: *mut WAVEFORMATEX = ptr::null_mut();
        let hr_supported = (*audio_client).IsFormatSupported(
            AUDCLNT_SHAREMODE_SHARED,
            mix_format,
            &mut closest_match,
        );

        let format_to_use = if hr_supported == S_OK {
            println!("Format supported directly.");
            print_audio_format_details(mix_format);
            mix_format
        } else if hr_supported == S_FALSE && !closest_match.is_null() {
            println!("Format not supported, using closest match.");
            print_audio_format_details(closest_match);
            closest_match
        } else {
            return Err("Unsupported format".into());
        };

        const REFTIMES_PER_100NS: REFERENCE_TIME = 10_000;
        let buffer_duration = 1000 * REFTIMES_PER_100NS;

        let hr_init = (*audio_client).Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            buffer_duration,
            0,
            format_to_use,
            ptr::null_mut(),
        );

        if hr_supported == S_FALSE && !closest_match.is_null() {
            CoTaskMemFree(closest_match as *mut _);
        }

        if FAILED(hr_init) {
            Err(format!(
                "Failed to initialize IAudioClient. HRESULT: 0x{:x}",
                hr_init
            )
            .into())
        } else {
            Ok(())
        }
    }
}
const KSDATAFORMAT_SUBTYPE_PCM: GUID = GUID {
    Data1: 0x00000001,
    Data2: 0x0000,
    Data3: 0x0010,
    Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
};

const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: GUID = GUID {
    Data1: 0x00000003,
    Data2: 0x0000,
    Data3: 0x0010,
    Data4: [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71],
};
fn is_guid_equal(guid1: &GUID, guid2: &GUID) -> bool {
    guid1.Data1 == guid2.Data1
        && guid1.Data2 == guid2.Data2
        && guid1.Data3 == guid2.Data3
        && guid1.Data4 == guid2.Data4
}
fn print_audio_format_details(format_ptr: *const WAVEFORMATEX) {
    if format_ptr.is_null() {
        println!("Error: format_ptr is null.");
        return;
    }

    unsafe {
        let format_ref = &*format_ptr;
        let sample_rate = format_ref.nSamplesPerSec;
        let bit_depth = format_ref.wBitsPerSample;
        let channels = format_ref.nChannels;
        let format_tag = format_ref.wFormatTag;
        println!("Sample Rate: {}", sample_rate);
        println!("Bit Depth: {}", bit_depth);
        println!("Channels: {}", channels);

        if format_tag == WAVE_FORMAT_PCM {
            println!("Format: PCM");
        } else if format_tag == WAVE_FORMAT_EXTENSIBLE {
            let extensible_format = &*(format_ptr as *const WAVEFORMATEXTENSIBLE);
            let sub_format = extensible_format.SubFormat;
            if is_guid_equal(&sub_format, &KSDATAFORMAT_SUBTYPE_PCM) {
                println!("Format: PCM (Extensible)");
            } else if is_guid_equal(&sub_format, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
                println!("Format: IEEE Float (Extensible)");
            } else {
                println!("Format: Unknown Extensible Format");
            }
            let samples = extensible_format.Samples;
            let channel_mask = extensible_format.dwChannelMask;
            println!("Samples: {}", samples);
            println!("Channel Mask: {}", channel_mask);
        } else {
            println!("Format: Unknown");
        }
    }
}
fn get_capture_client(
    audio_client: *mut IAudioClient,
) -> Result<*mut IAudioCaptureClient, Box<dyn Error>> {
    let mut capture_client: *mut IAudioCaptureClient = std::ptr::null_mut();
    let hr_service = unsafe {
        (*audio_client).GetService(
            &IAudioCaptureClient::uuidof(),
            &mut capture_client as *mut _ as *mut _,
        )
    };
    if FAILED(hr_service) {
        Err(format!("Failed to get capture client. HRESULT: 0x{:x}", hr_service).into())
    } else {
        Ok(capture_client)
    }
}

fn start_audio_client(audio_client: *mut IAudioClient) -> Result<(), Box<dyn Error>> {
    let hr_start = unsafe { (*audio_client).Start() };

    if FAILED(hr_start) {
        Err(format!("Failed to start capturing audio. HRESULT: 0x{:x}", hr_start).into())
    } else {
        Ok(())
    }
}

fn capture_audio_loop(
    capture_client: *mut IAudioCaptureClient,
    tx: Sender<Vec<u8>>,
    circular_buffer_queue: Arc<SegQueue<CircularBuffer>>,
) -> Result<(), Box<dyn Error>> {
    let mut circular_buffer = CircularBuffer::new(44100 * 4);
    loop {
        let mut buffer: *mut u8 = ptr::null_mut();
        let mut frames_available: u32 = 0;
        let mut flags: u32 = 0;
        let hr_buffer = unsafe {
            (*capture_client).GetBuffer(
                &mut buffer,
                &mut frames_available,
                &mut flags,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        if hr_buffer == AUDCLNT_E_DEVICE_INVALIDATED {
            println!("Device invalidated");
            break;
        } else if hr_buffer == AUDCLNT_S_BUFFER_EMPTY {
            continue;
        } else if hr_buffer != 0 {
            println!("Unexpected HRESULT: {:X}", hr_buffer);
            break;
        }

        let samples = if flags & AUDCLNT_BUFFERFLAGS_SILENT != 0 {
            insert_silent_frames_with_timing(
                tx.clone(),
                44100,
                calculate_duration(frames_available, 44100),
            )
        } else {
            convert_to_u8(buffer, frames_available)
        };
        circular_buffer.write(&samples);

        if circular_buffer.ready_to_send() {
            let mut data_to_send = Vec::new();
            circular_buffer.read_to_vec(&mut data_to_send);

            match tx.send(data_to_send.clone()) {
                Ok(_) => {
                    println!("Sent audio data");
                    circular_buffer.reset();
                }
                Err(e) => {
                    eprintln!("Failed to send audio data: {}", e);
                    break;
                }
            }

            circular_buffer = circular_buffer_queue
                .pop()
                .unwrap_or_else(|| CircularBuffer::new(44100 * 4));
        }

        let hr_release = unsafe { (*capture_client).ReleaseBuffer(frames_available) };
        if hr_release != 0 {
            break;
        }
    }
    Ok(())
}

fn calculate_duration(frames: u32, sample_rate: u32) -> Duration {
    let duration = frames as f64 / sample_rate as f64;
    Duration::from_secs_f64(duration)
}

fn insert_silent_frames_with_timing(
    tx: Sender<Vec<u8>>,
    sample_rate: u32,
    duration: Duration,
) -> Vec<u8> {
    let num_bytes = (sample_rate as f64 * duration.as_secs_f64()) as usize * 8;
    let silent_frames = vec![0u8; num_bytes];
    tx.send(silent_frames.clone()).unwrap();
    silent_frames
}

fn convert_to_u8(buffer: *mut u8, frames_available: u32) -> Vec<u8> {
    let mut samples = Vec::with_capacity(frames_available as usize * 8);
    unsafe {
        let buffer = std::slice::from_raw_parts(buffer, frames_available as usize * 8);
        samples.extend_from_slice(buffer);
    }
    samples
}

fn release_resources(audio_client: *mut IAudioClient) {
    unsafe {
        (*audio_client).Release();
    }
}

fn get_default_audio_device(role: ERole) -> Option<*mut IMMDevice> {
    unsafe {
        let mut enumerator: *mut IMMDeviceEnumerator = std::ptr::null_mut();
        if winapi::um::combaseapi::CoCreateInstance(
            &winapi::um::mmdeviceapi::MMDeviceEnumerator::uuidof(),
            std::ptr::null_mut(),
            winapi::um::combaseapi::CLSCTX_ALL,
            &winapi::um::mmdeviceapi::IMMDeviceEnumerator::uuidof(),
            &mut enumerator as *mut _ as *mut _,
        ) == 0
        {
            let mut device: *mut IMMDevice = std::ptr::null_mut();
            if (*enumerator).GetDefaultAudioEndpoint(
                winapi::um::mmdeviceapi::eRender,
                role as u32,
                &mut device,
            ) == 0
            {
                let friendly_name = get_device_friendly_name(device);
                if let Some(name) = friendly_name {
                    println!("Default Audio Render Device: {}", name);
                } else {
                    println!("Failed to get device friendly name");
                }
                return Some(device);
            }
        }
    }
    None
}

fn get_device_friendly_name(device: *mut IMMDevice) -> Option<String> {
    unsafe {
        let mut property_store = std::ptr::null_mut();
        if (*device).OpenPropertyStore(0, &mut property_store) != S_OK {
            return None;
        }

        let mut variant: PROPVARIANT = std::mem::zeroed();
        if (*property_store).GetValue(
            &winapi::um::functiondiscoverykeys_devpkey::PKEY_Device_FriendlyName,
            &mut variant,
        ) != S_OK
        {
            return None;
        }

        if variant.vt != winapi::shared::wtypes::VT_LPWSTR as u16 {
            return None;
        }

        let friendly_name = {
            let pwsz = *variant.data.pwszVal();
            let len = (0..).take_while(|&i| *pwsz.offset(i) != 0).count();
            OsString::from_wide(&std::slice::from_raw_parts(pwsz, len))
                .to_string_lossy()
                .into_owned()
        };

        (*property_store).Release();
        Some(friendly_name)
    }
}

fn get_audio_device_details(
    format_ptr: *const WAVEFORMATEX,
    rx: mpsc::Sender<Vec<u8>>,
) -> Result<(), &'static str> {
    if format_ptr.is_null() {
        return Err("format_ptr is null");
    }

    unsafe {
        let format_ref = &*format_ptr;
        let sample_rate = format_ref.nSamplesPerSec;
        let bit_depth = format_ref.wBitsPerSample;
        let channels = format_ref.nChannels;
        let mut details = Vec::new();
        details.extend_from_slice(&sample_rate.to_le_bytes());
        details.extend_from_slice(&bit_depth.to_le_bytes());
        details.extend_from_slice(&channels.to_le_bytes());

        rx.send(details).unwrap();
    }
    Ok(())
}
fn send_audio_data(
    rx: mpsc::Receiver<Vec<u8>>,
    rx_device: mpsc::Receiver<Vec<u8>>,
    address: &str,
) -> Result<(), Box<dyn Error>> {
    let mut stream = TcpStream::connect(address)?;

    let batch_size = 8192;
    while let Ok(details) = rx_device.recv() {
        if details.len() < 8 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Incomplete device details",
            )));
        }

        let sample_rate = u32::from_le_bytes([details[0], details[1], details[2], details[3]]);
        let bit_depth = u16::from_le_bytes([details[4], details[5]]);
        let channels = u16::from_le_bytes([details[6], details[7]]);
        let device = DeviceDetails {
            sample_rate,
            bit_depth,
            channels,
        };
        println!("Sending device details: {:?}", device);
        let serialized = serde_json::to_string(&device)?;
        stream.write_all(serialized.as_bytes())?;
        stream.write_all(b"\n")?;
        stream.flush()?;
    }

    while let Ok(audio_data) = rx.recv() {
        if audio_data.is_empty() {
            eprintln!("Received empty audio data");
            break;
        }

        for chunk in audio_data.chunks(batch_size) {
            stream.write_all(chunk)?;
        }
    }

    Ok(())
}

// fn give_audio_output(receiver: mpsc::Receiver<Vec<u8>>) -> Result<(), Box<dyn Error>> {
//     let spec = hound::WavSpec {
//         channels: 2,
//         sample_rate: 44100,
//         bits_per_sample: 32,
//         sample_format: hound::SampleFormat::Float,
//     };

//     let mut writer = hound::WavWriter::create("output.wav", spec)?;

//     while let Ok(audio_data) = receiver.recv() {
//         for chunk in audio_data.chunks(4) {
//             let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
//             writer.write_sample(sample)?;
//         }
//     }

//     writer.finalize()?;

//     Ok(())
// }

fn main() {
    unsafe {
        if CoInitializeEx(std::ptr::null_mut(), COINITBASE_MULTITHREADED) != S_OK {
            eprintln!("COM initialization failed");
            return;
        }
        let (tx, rx) = mpsc::channel();
        let (tx_device, rx_device) = mpsc::channel();
        let circular_buffer = Arc::new(SegQueue::new());

        let handle_capture = thread::spawn(move || {
            let wave_format = create_wave_format(48000, 32, 2);
            if wave_format.is_null() {
                eprintln!("Failed to create wave format");
                return;
            }

            let device =
                get_default_audio_device(eCapture).expect("Failed to get default audio device");
            let audio_client =
                activate_audio_client(device).expect("Failed to activate audio client");
            let mix_format = get_mix_format(audio_client).expect("Failed to get mix format");
            let _device_details = get_audio_device_details(mix_format, tx_device)
                .expect("Failed to get audio device details");

            if let Err(err) = initialize_audio_client(audio_client, mix_format) {
                eprintln!("Failed to initialize audio client: {}", err);
                CoTaskMemFree(wave_format as *mut _);
                release_resources(audio_client);
                return;
            }

            CoTaskMemFree(mix_format as *mut _);

            let capture_client = match get_capture_client(audio_client) {
                Ok(client) => client,
                Err(err) => {
                    eprintln!("Failed to get capture client: {}", err);
                    CoTaskMemFree(wave_format as *mut _);
                    release_resources(audio_client);
                    return;
                }
            };

            if let Err(err) = start_audio_client(audio_client) {
                eprintln!("Failed to start audio client: {}", err);
                release_resources(audio_client);
                return;
            }

            if let Err(err) = capture_audio_loop(capture_client, tx, circular_buffer.clone()) {
                eprintln!("Failed to capture audio: {}", err);
                release_resources(audio_client);
                return;
            }

            release_resources(audio_client);
            CoTaskMemFree(wave_format as *mut _);
        });

        let handle_tcp = thread::spawn(move || {
            if let Err(err) = send_audio_data(rx, rx_device, "192.168.1.27:8000") {
                eprintln!("Failed to send audio data: {}", err);
            }
        });

        if let Err(err) = handle_tcp.join() {
            eprintln!("Failed to join TCP thread: {:?}", err);
        } else {
            println!("TCP thread joined successfully");
        }
        if let Err(err) = handle_capture.join() {
            eprintln!("Failed to join capture thread: {:?}", err);
        } else {
            println!("Capture thread joined successfully");
        }

        CoUninitialize();
    }
}


