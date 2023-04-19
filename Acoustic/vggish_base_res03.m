function [vggish_base]=vggish_base_res03(x,fs)

dur=3;%3s 
au_length=fs*dur;
if length(x)<au_length
    x=[x;zeros( au_length-length(x) , 1 )];
    vggish_base = vggishFeatures(x,fs,'OverlapPercentage',90);
else
    vggish_base = vggishFeatures(x,fs,'OverlapPercentage',90);
end

end