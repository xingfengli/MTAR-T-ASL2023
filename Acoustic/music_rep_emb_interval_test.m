function [musicV2_Interval]=music_rep_emb_interval_test(musicV2,musicV2_melody)
%close all;
%clear all;
%audio loading
% [x,fs]=audioread('A08a05Nb.wav');

for m=1:size(musicV2,2)
    f_syn=musicV2(:,m);
    f_mel=musicV2_melody(:,m);
    tmp_pref=4*(10^(-10));
    TMP=[];
    for s=1:length(f_syn)
        tmp_bfr=(10^(f_syn(s)/10))*4*(10^(-10));
        tmp_aft=(10^(f_mel(s)/10))*4*(10^(-10));
        tmp_dB=10*log10((tmp_bfr+tmp_aft)/tmp_pref); 
        TMP=[TMP;tmp_dB];
    end
    musicV2_Interval(:,m)=TMP;
end
melcc_Interval = cepstralCoefficients(abs(musicV2_Interval),'NumCoeffs',13);
melccDelta_Interval = audioDelta(melcc_Interval);
melccDeltaDelta_Interval = audioDelta(melccDelta_Interval);
musicV2_Interval=[ melcc_Interval melccDelta_Interval melccDeltaDelta_Interval ];
end