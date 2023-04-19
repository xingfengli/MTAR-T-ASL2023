function [musicV2]=music_rep_emb_test(x,fs)
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

for t=1:size(spec,2)
frame_mel=spec(:,t);
mus_rep_tmp=zeros(length(frame_mel),length(frame_mel));

p_ref=4*(10^(-10));
for f=1:length(frame_mel)
    p_prior=(10^(frame_mel(f)/10))*4*(10^(-10));
    for s=f:length(frame_mel)
        p_follow=(10^(frame_mel(s)/10))*4*(10^(-10));
        mus_rep_tmp(f,s)=10*log10((p_prior+p_follow)/p_ref);
        mus_rep_tmp(s,f)=10*log10((p_prior+p_follow)/p_ref); 
    end  
end

%musicV(:,:,t)=mus_rep_tmp;
musicV2_tmp=[];
for l=1:size(mus_rep_tmp,1)
    triu_mus=triu(mus_rep_tmp);
    musicV2_tmp=[musicV2_tmp;triu_mus(l,l:end)'];
end
% musicV2(:,t)=musicV2_tmp(find(musicV2_tmp~=0));
musicV2(:,t)=musicV2_tmp;
end
% 
% figure;subplot(2,1,1);imagesc(spec);
% subplot(2,1,2);imagesc(musicV2);
% 
% melcc = cepstralCoefficients(abs(musicV2),'NumCoeffs',13);
% melccDelta = audioDelta(melcc);
% melccDeltaDelta = audioDelta(melccDelta);
% musicV2=[melcc melccDelta melccDeltaDelta];
% 
% figure;imagesc(musicV2(:,2:end))
end