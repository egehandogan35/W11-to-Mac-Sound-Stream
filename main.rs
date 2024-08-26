use coreaudio::audio_unit::audio_format::LinearPcmFlags;
use coreaudio::audio_unit::macos_helpers::get_default_device_id;
use coreaudio::audio_unit::render_callback::data::Interleaved;
use coreaudio::audio_unit::render_callback::Args;
use coreaudio::audio_unit::{AudioUnit, Element, IOType, SampleFormat, Scope, StreamFormat};
use std::f32::consts::PI;
use std::io::Read;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::thread;

#[derive(serde::Deserialize)]
struct DeviceDetails {
    sample_rate: u32,
    bit_depth: u16,
    channels: u16,
}
struct CircularBuffer {
    buffer: Vec<f32>,
    read_pos: usize,
    write_pos: usize,
    size: usize,
    is_full: bool,
}

impl CircularBuffer {
    fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            read_pos: 0,
            write_pos: 0,
            size,
            is_full: false,
        }
    }

    fn write(&mut self, data: &[f32]) {
        for &sample in data {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.size;
            if self.is_full {
                self.read_pos = (self.read_pos + 1) % self.size;
            }
            self.is_full = self.write_pos == self.read_pos;
        }
    }

    fn read(&mut self, num_samples: usize) -> Vec<f32> {
        let mut output = vec![0.0; num_samples];
        for i in 0..num_samples {
            if self.read_pos != self.write_pos || self.is_full {
                output[i] = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.size;
                self.is_full = false;
            } else {
                break;
            }
        }
        output
    }
}


struct LowPassFilter {
    alpha: f32,
    prev_output: f32,
}

impl LowPassFilter {
    fn new(cutoff_frequency: f32, sample_rate: f32) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * PI * cutoff_frequency);
        let alpha = dt / (rc + dt);

        LowPassFilter {
            alpha,
            prev_output: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.alpha * input + (1.0 - self.alpha) * self.prev_output;
        self.prev_output = output;
        output
    }
}

struct HighPassFilter {
    alpha: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighPassFilter {
    fn new(cutoff_frequency: f32, sample_rate: f32) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * PI * cutoff_frequency);
        let alpha = rc / (rc + dt);

        HighPassFilter {
            alpha,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.alpha * (self.prev_output + input - self.prev_input);
        self.prev_input = input;
        self.prev_output = output;
        output
    }
}



struct BassCompressor {
    threshold: f32,
    ratio: f32,
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
    gain: f32,
}

impl BassCompressor {
    fn new(threshold: f32, ratio: f32, attack_time: f32, release_time: f32, sample_rate: f32) -> Self {
        let attack_coeff = 1.0 - (-1.0 / (attack_time * sample_rate)).exp();
        let release_coeff = 1.0 - (-1.0 / (release_time * sample_rate)).exp();
        BassCompressor {
            threshold,
            ratio,
            attack_coeff,
            release_coeff,
            envelope: 0.0,
            gain: 1.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let abs_input = input.abs();
        if abs_input > self.envelope {
            self.envelope += self.attack_coeff * (abs_input - self.envelope);
        } else {
            self.envelope += self.release_coeff * (abs_input - self.envelope);
        }

        if self.envelope > self.threshold {
            self.gain = self.threshold + (self.envelope - self.threshold) / self.ratio;
            self.gain = self.threshold / self.gain;
        } else {
            self.gain = 1.0;
        }

        input * self.gain
    }
}

struct EnhancedLimiter {
    threshold: f32,
    release_coeff: f32,
    envelope: f32,
}

impl EnhancedLimiter {
    fn new(threshold: f32, release_time: f32, sample_rate: f32) -> Self {
        let release_coeff = 1.0 - (-1.0 / (release_time * sample_rate)).exp();
        EnhancedLimiter {
            threshold,
            release_coeff,
            envelope: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let abs_input = input.abs();
        if abs_input > self.envelope {
            self.envelope = abs_input;
        } else {
            self.envelope += self.release_coeff * (abs_input - self.envelope);
        }

        if self.envelope > self.threshold {
            self.threshold * input.signum()
        } else {
            input
        }
    }
}



struct ParametricEqualizer {
    low_shelf_gain: f32,
    mid_gain: f32,
    high_shelf_gain: f32,
}

impl ParametricEqualizer {
    fn new(low_shelf_gain: f32, mid_gain: f32, high_shelf_gain: f32) -> Self {
        ParametricEqualizer {
            low_shelf_gain,
            mid_gain,
            high_shelf_gain,
        }
    }

    fn process(&self, sample: f32) -> f32 {
        let low_shelf = sample * self.low_shelf_gain;
        let mid = sample * self.mid_gain;
        let high_shelf = sample * self.high_shelf_gain;
        (low_shelf + mid + high_shelf) / 3.0
    }
}
struct AmplitudeLimiter {
    max_amplitude: f32,
}

impl AmplitudeLimiter {
    fn new(max_amplitude: f32) -> Self {
        AmplitudeLimiter { max_amplitude }
    }

    fn process(&self, input: f32) -> f32 {
        if input.abs() > self.max_amplitude {
            input.signum() * self.max_amplitude
        } else {
            input
        }
    }
}


fn u8_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| {
        let u32_value = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        f32::from_bits(u32_value)
    }).collect()
}


fn get_default_audio_device_mac() -> AudioUnit {
    let audio_unit = match AudioUnit::new(IOType::DefaultOutput) {
        Ok(unit) => unit,
        Err(e) => {
            eprintln!("Failed to create AudioUnit: {}", e);
            std::process::exit(1);
        }
    };
    println!("audio unit sample rate: {:?}", audio_unit.sample_rate());

    audio_unit
}

fn clamp_audio_data(data: &mut [f32], min_value: f32, max_value: f32) {
    for sample in data.iter_mut() {
        if *sample < min_value {
            *sample = min_value;
        } else if *sample > max_value {
            *sample = max_value;
        }
    }
}

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        (x * PI).sin() / (x * PI)
    }
}

fn hamming_window(n: usize, window_size: usize) -> f32 {
    0.54 - 0.46 * ((2.0 * PI * n as f32) / (window_size as f32 - 1.0)).cos()
}

fn resample_audio(
    input_samples: &[f32],
    input_rate: f32,
    output_rate: f32,
    input_channels: u16,
    output_channels: u16,
) -> Vec<f32> {
    let resample_ratio = output_rate / input_rate;
    let output_length = ((input_samples.len() / input_channels as usize) as f32 * resample_ratio).ceil() as usize * output_channels as usize;
    let mut output_samples = Vec::with_capacity(output_length);

    let window_size = 16;
    let cutoff_frequency = if output_rate < input_rate {
        output_rate / 2.0 
    } else {
        input_rate / 2.0 
    };
    let attack_time = 0.01;
    let release_time = 0.1;
    let compression_threshold = 0.1;
    let compression_ratio = 2.0;
    let limiter_threshold = 0.9;

    let mut low_pass_filter = LowPassFilter::new(cutoff_frequency, input_rate);
    let mut high_pass_filter = HighPassFilter::new(cutoff_frequency, output_rate);
    let mut bass_compressor = BassCompressor::new(compression_threshold, compression_ratio, attack_time, release_time, output_rate);
    let mut enhanced_limiter = EnhancedLimiter::new(limiter_threshold, release_time, output_rate);
    let parametric_equalizer = ParametricEqualizer::new(1.0, 1.0, 1.0);
    let amplitude_limiter = AmplitudeLimiter::new(0.99);

    for i in 0..output_length / output_channels as usize {
        let input_index = i as f32 / resample_ratio;
        let input_index_int = input_index.floor() as isize;
        let input_index_frac = input_index - input_index.floor();
        let mut sample = vec![0.0; input_channels as usize];
        let mut sum_weights = 0.0;

        for n in -window_size..=window_size {
            let idx = input_index_int + n;
            if idx >= 0 && (idx as usize) < input_samples.len() / input_channels as usize {
                let sinc_val = sinc(n as f32 - input_index_frac);
                let window_val = hamming_window(
                    (n + window_size) as usize,
                    (2 * window_size + 1).try_into().unwrap(),
                );
                let weight = sinc_val * window_val;
                for ch in 0..input_channels {
                    sample[ch as usize] += input_samples[(idx as usize * input_channels as usize) + ch as usize] * weight;
                }
                sum_weights += weight;
            }
        }
        if sum_weights != 0.0 {
            for ch in 0..input_channels {
                sample[ch as usize] /= sum_weights;
            }
        }

        for ch in 0..output_channels {
            let mut processed_sample = if ch < input_channels {
                sample[ch as usize]
            } else {
                0.0 
            };
            if output_rate < input_rate {
                processed_sample = low_pass_filter.process(processed_sample);
            } else {
                processed_sample = high_pass_filter.process(processed_sample);
            }
            processed_sample = bass_compressor.process(processed_sample);
            processed_sample = enhanced_limiter.process(processed_sample);
            processed_sample = parametric_equalizer.process(processed_sample);
            processed_sample = amplitude_limiter.process(processed_sample);
            clamp_audio_data(&mut [processed_sample], -1.0, 1.0);
            output_samples.push(processed_sample);
        }
    }

    output_samples
}

fn send_to_audio_device(rx: mpsc::Receiver<Option<Vec<f32>>>, buffer: Arc<Mutex<CircularBuffer>>) {
    let mut audio_unit = get_default_audio_device_mac();

    const SAMPLE_FORMAT: SampleFormat = SampleFormat::F32;
    let format_flag = match SAMPLE_FORMAT {
        SampleFormat::F32 => LinearPcmFlags::IS_FLOAT,
        SampleFormat::I32 | SampleFormat::I16 | SampleFormat::I8 => {
            LinearPcmFlags::IS_SIGNED_INTEGER
        }
        _ => {
            unimplemented!("Other formats are not implemented for this example.");
        }
    };

    let stream_format = StreamFormat {
        sample_rate: 44100.0,
        sample_format: SampleFormat::F32,
        channels: 2,
        flags: format_flag | LinearPcmFlags::IS_PACKED | LinearPcmFlags::IS_FLOAT,
    };

    let asbd = stream_format.to_asbd();
    let id = get_default_device_id(false).unwrap();
    let _ = audio_unit.set_property(id, Scope::Input, Element::Output, Some(&asbd));

    audio_unit.set_stream_format(stream_format, Scope::Input).unwrap();

    let buffer_clone = Arc::clone(&buffer);
    let render_callback = Box::new(move |args: Args<Interleaved<f32>>| {
        let num_frames = args.num_frames * 2;
        let audio_data = buffer.lock().unwrap().read(num_frames);
   
        let mut data = args.data;
        let buffer = &mut data.buffer;
    
        for i in 0..num_frames {
            buffer[i] = audio_data[i];
        }
  

        Ok(())
    });

    match audio_unit.set_render_callback(render_callback) {
        Ok(_) => println!("Render callback set successfully"),
        Err(e) => {
            eprintln!("Failed to set render callback: {}", e);
            std::process::exit(1);
        }
    }

    match audio_unit.start() {
        Ok(_) => println!("Audio unit started successfully"),
        Err(e) => {
            eprintln!("Failed to start audio unit: {}", e);
            std::process::exit(1);
        }
    }

    while let Ok(data) = rx.recv() {
        if let Some(samples) = data {
            let mut buf = buffer_clone.lock().unwrap();
            buf.write(&samples);
        } else {
            break;
        }
    }

    println!("send_to_audio_device end");
}


fn main() {
    let server = match TcpListener::bind("0.0.0.0:8000") {
        Ok(server) => {
            println!("Server started on port 8000");
            server
        },
        Err(err) => {
            eprintln!("Error: {}", err);
            std::process::exit(1);
        }
    };

    let (tx, rx) = mpsc::channel();
    let buffer = Arc::new(Mutex::new(CircularBuffer::new(44100))); 

    let buffer_clone = Arc::clone(&buffer);
    thread::spawn(move || {
        send_to_audio_device(rx, buffer_clone);
    });

    let output_sample_rate = 44100.0;
    let output_channels = 2;
    let mut buffer = [0; 8192]; 

    for stream in server.incoming() {
        match stream {
            Ok(mut stream) => {
                println!("New connection: {}", stream.peer_addr().unwrap());
                match stream.read(&mut buffer) {
                    Ok(bytes_read) => {
                        let details_str = String::from_utf8_lossy(&buffer[..bytes_read]);
                        println!("Received device details: {}", details_str);
                        if let Ok(device_details) = serde_json::from_str::<DeviceDetails>(&details_str) {
                            println!("Received device details: samplerate={}, bit_depth={}, channels={}",
                                      device_details.sample_rate, device_details.bit_depth, device_details.channels);

                            if device_details.sample_rate as f32 != output_sample_rate {
                                println!("Resampling is needed.");
                                while let Ok(bytes_read) = stream.read(&mut buffer) {
                                    if bytes_read == 0 {
                                        break;
                                    }

                                    let audio_data = u8_to_f32(&buffer[..bytes_read]);
                                    let resampled_audio_data = resample_audio(&audio_data, device_details.sample_rate as f32, output_sample_rate, device_details.channels, output_channels);
                                    let _ = tx.send(Some(resampled_audio_data));
                                }
                            } else {
                                println!("Resampling is not needed.");
                                while let Ok(bytes_read) = stream.read(&mut buffer) {
                                    if bytes_read == 0 {
                                        break;
                                    }

                                    let audio_data = u8_to_f32(&buffer[..bytes_read]);
                                    let _ = tx.send(Some(audio_data));
                                }
                            }
                        } else {
                            eprintln!("Failed to parse device details");
                        }
                    },
                    Err(e) => {
                        eprintln!("Failed to read device details: {}", e);
                    }
                }
                let _ = tx.send(None);
            },
            Err(e) => eprintln!("Failed to accept connection: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_to_f32() {
        let bytes: [u8; 8] = [0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0xBF];
        let result = u8_to_f32(&bytes);
        assert_eq!(result, vec![1.0, -1.0]);
    }
}
