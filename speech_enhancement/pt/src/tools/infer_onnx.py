#!/usr/bin/env python3
"""Inferenza offline di un file WAV usando un modello ONNX.

Input: noisy wav
Output: denoised wav + spectrogrammi (noisy e denoised)

Requisiti: onnxruntime, librosa, soundfile, numpy, matplotlib
"""
import argparse
import os

from matplotlib.path import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
import librosa
import librosa.display
import matplotlib.pyplot as plt

ENV = os.path.dirname(os.path.abspath(__file__))
INPUT_AUDIO_DIR = os.path.join(ENV, 'input')
OUTPUT_AUDIO_DIR = os.path.join(ENV, 'output')
SPECTROGRAM_DIR = os.path.join(OUTPUT_AUDIO_DIR, 'img')
ONNX_MODEL_DIR = os.path.join(ENV, 'onnx')


class Inferencer():
    def __init__(self, in_audio_dir = INPUT_AUDIO_DIR, out_audio_dir = OUTPUT_AUDIO_DIR, model_dir = ONNX_MODEL_DIR,  spectrogram_dir = SPECTROGRAM_DIR, sr=16000, n_fft=512, hop_length=160, win_length=320, win_length_static=40):
        self.in_audio_dir = in_audio_dir
        self.out_audio_dir = out_audio_dir
        self.model_dir = model_dir
        self.spectrogram_dir = spectrogram_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_length_static = win_length_static
        
        self._mkdir()
        
        self.audio_files_list: list[Path] = self._get_audio_file_list()
        self.onnx_files_list: list[Path] = self._get_onnx_file_list()
        
        self._run_inference()
    
    
    def _mkdir(self, path=None):
        if path is None:
            os.makedirs(INPUT_AUDIO_DIR, exist_ok=True)
            os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
            os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
            os.makedirs(ONNX_MODEL_DIR, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            
    def _get_audio_file_list(self):
        audio_files = []
        for file_name in os.listdir(self.in_audio_dir):
            if file_name.lower().endswith('.wav'):
                audio_files.append(os.path.join(self.in_audio_dir, file_name))
            else:
                raise RuntimeError("Unsupported file format.")
        
        assert len(audio_files) > 0, f"No WAV files found in {self.in_audio_dir}"
        return audio_files

    def _get_onnx_file_list(self):
        onnx_files = []
        for file_name in os.listdir(self.model_dir):
            if file_name.lower().endswith('.onnx'):
                onnx_files.append(os.path.join(self.model_dir, file_name))
            else:
                raise RuntimeError("Unsupported file format.")
        
        assert len(onnx_files) > 0, f"No ONNX files found in {self.model_dir}"
        return onnx_files

    def _load_audio(self, file_path):
        audio, sr = sf.read(file_path)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
           
        return audio

    def _save_audio(self, path, audio, sr):
        # normalize if necessary
        maxv = np.max(np.abs(audio))
        if maxv > 1.0:
            audio = audio / maxv
        sf.write(path, audio, sr)

    def _make_spectrogram_image(self, S, sr, hop_length, out_path, title="Spectrogram"):
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _run_inference(self):
        for onnx_file in self.onnx_files_list:
            for audio_file in self.audio_files_list:
                print(f"Processing {os.path.basename(audio_file)} with model {os.path.basename(onnx_file)}...")
                
                audio_data = self._load_audio(audio_file)
                
                if audio_file.endswith('.static.wav') and self.win_length_static is not None:
                    print(f"Using static win_length {self.win_length_static} for {os.path.basename(audio_file)}")
                    self.win_length = self.win_length_static    
                # STFT
                S = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window='hann', center=True)
                mag = np.abs(S)
                phase = np.angle(S)

                # Prepare input for ONNX: (1, n_bins, T)
                inp = mag[np.newaxis, :, :].astype(np.float32)

                # Load ONNX
                sess = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
                input_name = sess.get_inputs()[0].name

                # Inspect model input shape to see if sequence axis is static
                input_shape = sess.get_inputs()[0].shape
                expected_T = None
                if len(input_shape) >= 3:
                    axis2 = input_shape[2]
                    if isinstance(axis2, int):
                        expected_T = axis2

                # If model accepts dynamic sequence length or it matches current mag, run directly
                if expected_T is None or expected_T == mag.shape[1]:
                    outputs = sess.run(None, {input_name: inp})
                    mask = outputs[0]
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                else:
                    # Model expects a fixed sequence length (e.g., 40). Process in chunks.
                    T = mag.shape[1]
                    masks = np.zeros_like(mag, dtype=np.float32)
                    chunk_len = expected_T
                    for start in range(0, T, chunk_len):
                        end = start + chunk_len
                        chunk = mag[:, start:end]
                        # pad last chunk if needed
                        if chunk.shape[1] < chunk_len:
                            pad_width = chunk_len - chunk.shape[1]
                            chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')

                        inp_chunk = chunk[np.newaxis, :, :].astype(np.float32)
                        out_chunk = sess.run(None, {input_name: inp_chunk})[0]
                        if out_chunk.ndim == 3 and out_chunk.shape[0] == 1:
                            out_chunk = out_chunk[0]

                        # try to transpose if needed
                        if out_chunk.shape != (mag.shape[0], chunk_len):
                            if out_chunk.shape[0] == chunk_len and out_chunk.shape[1] == mag.shape[0]:
                                out_chunk = out_chunk.T
                            else:
                                raise RuntimeError(f"Model output chunk shape {out_chunk.shape} is incompatible with expected (n_bins, chunk_len)")

                        # trim padding for the last chunk and store
                        valid_len = min(chunk_len, max(0, T - start))
                        masks[:, start:start+valid_len] = out_chunk[:, :valid_len]

                    mask = masks

                # Apply magnitude or complex mask
                mask = mask.astype(np.float32)
                if mask.shape[0] == 2 * mag.shape[0]:
                    # Complex mask: (2F, T) -> Mr, Mi
                    mr = mask[:mag.shape[0], :]
                    mi = mask[mag.shape[0]:, :]
                    S_real = mag * np.cos(phase)
                    S_imag = mag * np.sin(phase)
                    S_deno_real = mr * S_real - mi * S_imag
                    S_deno_imag = mr * S_imag + mi * S_real
                    S_deno = S_deno_real + 1j * S_deno_imag
                    denoised_mag = np.abs(S_deno)
                elif mask.shape[0] == mag.shape[0]:
                    # Magnitude mask: (F, T)
                    denoised_mag = mag * mask
                    S_deno = denoised_mag * np.exp(1j * phase)
                else:
                    raise RuntimeError(
                        f"Model output shape {mask.shape} is incompatible with expected (F,T) or (2F,T)."
                    )
                y_deno = librosa.istft(S_deno, hop_length=self.hop_length, win_length=self.win_length, window='hann', length=len(audio_data))

                # Save audio
                # input wav file + model name + .wav
                input_name = os.path.splitext(os.path.basename(audio_file))[0]
                model_name = os.path.splitext(os.path.basename(onnx_file))[0]
                out_wav = os.path.join(self.out_audio_dir, f"{input_name}_{model_name}.wav")
                self._save_audio(out_wav, y_deno, self.sr)

                # Save spectrogram images
                noisy_img = os.path.join(self.spectrogram_dir, f'{input_name}.png')
                denoised_img = os.path.join(self.spectrogram_dir, f'{input_name}_{model_name}_den.png')
                self._make_spectrogram_image(mag, self.sr, self.hop_length, noisy_img, title='Noisy magnitude (dB)')
                self._make_spectrogram_image(denoised_mag, self.sr, self.hop_length, denoised_img, title='Denoised magnitude (dB)')

                print(f"Wrote denoised audio to: {os.path.basename(out_wav)}")
                print(f"Wrote spectrograms to: {os.path.basename(noisy_img)}, {os.path.basename(denoised_img)}")
                print()
        
    


def _ask(prompt, default=None, cast=str):
    if default is None:
        txt = f"{prompt}: "
    else:
        txt = f"{prompt} [{default}]: "
    val = input(txt).strip()
    if val == "":
        return default
    try:
        return cast(val)
    except Exception:
        print(f"Invalid value, expected type {cast.__name__}. Using default {default}.")
        return default


def parse_args():
    p = argparse.ArgumentParser(description='ONNX inference for STFT-TCNN style models (interactive)')
    p.add_argument('--n-fft', type=int, help='STFT n_fft', default=None)
    p.add_argument('--hop', type=int, help='STFT hop_length', default=None)
    p.add_argument('--win-length', type=int, help='STFT win_length', default=None)
    p.add_argument('--win-length-static', type=int, help='STFT win_length for static onnx', default=None)
    return p.parse_args()


def collect_inputs_interactive(args):
    n_fft = args.n_fft if args.n_fft is not None else _ask('STFT n_fft', default=512, cast=int)
    hop = args.hop if args.hop is not None else _ask('STFT hop_length', default=160, cast=int)
    win_length = args.win_length if args.win_length is not None else _ask('STFT win_length', default=320, cast=int)
    win_length_static = args.win_length_static if args.win_length_static is not None else _ask('STFT win_length for static onnx', default=40, cast=int)
    return dict(n_fft=n_fft, hop=hop, win_length=win_length, win_length_static=win_length_static)


def main():
    args = parse_args()
    params = collect_inputs_interactive(args)
    Inferencer(n_fft=params['n_fft'], hop_length=params['hop'], win_length=params['win_length'], win_length_static=params['win_length_static'])


if __name__ == '__main__':
    main()
