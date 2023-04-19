function [musicV2_melody]=music_rep_emb_melody_test(x,fs)
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

spec = 10*log10(spec+eps);

for prior=1:size(spec,2)
    beforeV=spec(:,prior);
    if prior~=size(spec,2)
        afterV=spec(:,prior+1);
    else
        afterV=spec(:,prior);
    end
    Melody_tmp=zeros(length(beforeV),length(afterV));
    tmp_pref=4*(10^(-10));
    for s=1:length(beforeV)
        for m=1:length(afterV)
            tmp_base=(10^(beforeV(s)/10))*4*(10^(-10));
            tmp_var=(10^(afterV(m)/10))*4*(10^(-10));
            Melody_tmp(s,m)=10*log10((tmp_base+tmp_var)/tmp_pref); 
        end
    end
    MelodyV=[];
    for p=1:size(Melody_tmp,1)
        for q=p:size(Melody_tmp,1)
            tmp_up=(10^(Melody_tmp(p,q)/10))*4*(10^(-10));
            tmp_down=(10^(Melody_tmp(q,p)/10))*4*(10^(-10));
            tmp_dB=10*log10((tmp_up+tmp_down)/tmp_pref); 
            MelodyV=[MelodyV; tmp_dB];
        end
    end
    musicV2_melody(:,prior)=MelodyV;
 
end
% musicV2_melody

% melcc_melody = cepstralCoefficients(abs(musicV2_melody),'NumCoeffs',13);
% melccDelta_melody = audioDelta(melcc_melody);
% melccDeltaDelta_melody = audioDelta(melccDelta_melody);
% musicV2_melody=[melcc_melody melccDelta_melody melccDeltaDelta_melody];

end