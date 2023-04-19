function [mfcc_base]=mfcc_base_res02(x,fs)
%close all;
%clear all;
%audio loading
% [x,fs]=audioread('A08a05Nb.wav');
windowLength = 640;
samplesPerHop = 160;
samplesOverlap = windowLength - samplesPerHop;

afe = audioFeatureExtractor('SampleRate',fs, ...
                            'Window',hann(windowLength,'periodic'), ...
                            'OverlapLength',samplesOverlap , ...
                            'mfcc',true, ...
                            'mfccDelta', true, ...
                            'mfccDeltaDelta',true); 
                        
mfcc_base = extract(afe,x); 
end