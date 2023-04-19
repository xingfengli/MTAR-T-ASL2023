function [spectr_base]=spectr_base_res04(x)

[s,~,~,~] = spectrogram(x,640,480,256);
spO=10*log10(abs(s)+eps);

spec_base=spO.';
% specDelta = audioDelta(spec_base);
% specDeltaDelta = audioDelta(specDelta);
% spec_PlusDelta=[ spec_base specDelta specDeltaDelta ];
% spectr_base=spec_PlusDelta;
spectr_base=spec_base;
end