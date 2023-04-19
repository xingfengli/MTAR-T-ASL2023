function [mel_base]=mel_base_res01(x,fs)
%close all;
%clear all;
%audio loading
% [x,fs]=audioread('A08a05Nb.wav');
windowLength = 640;
samplesPerHop = 160;
samplesOverlap = windowLength - samplesPerHop;
fftLength = 2*windowLength;
% numBands = 32;
numBands = 40;
% numBands = 512;

spec = melSpectrogram(x,fs, ...
    'Window',hamming(windowLength,'periodic'), ...
    'OverlapLength',samplesOverlap, ...
    'FFTLength',fftLength, ...
    'NumBands',numBands);

mel_base = 10*log10(spec+eps);
mel_base=mel_base.';
melDelta = audioDelta(mel_base);
melDeltaDelta = audioDelta(melDelta);
mel_PlusDelta=[ mel_base melDelta melDeltaDelta ];
mel_base=mel_PlusDelta;
end