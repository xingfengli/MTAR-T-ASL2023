function iemocap_9946au_80dim_ftr_extract_ieee2021_baselines4

agudir='/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/';
audir=dir([agudir,'allAu_SyTag_r/*.wav']);
au_names={audir.name}';
wav_sourcePath='/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/allAu_SyTag_r/';
ftrDir='/Users/xingfengli/Documents/RESEARCH/MusicRep/base4_csvSpec_129/';

for au=1:length(au_names)

%% loading audio file
    [x,fs]=audioread([wav_sourcePath,au_names{au}]);
    
 	[audio_in,audioNames,~]=eng_audioInfo_pre_loading(x);
	x=audio_in;
%% Proposed @2021
%% melSpectrogeam baseline-1:
%     [mel_base]=mel_base_res01(x,fs);
%     writematrix(mel_base.',[ftrDir,audioNames,'.csv']);
    
    %% mfcc baseline-2:
%     [mfcc_base]=mfcc_base_res02(x,fs);
%     writematrix(mfcc_base.',[ftrDir,audioNames,'.csv']);

    %% VGGish baseline-3:
%     [vggish_base]=vggish_base_res03(x,fs);
%     writematrix(vggish_base.',[ftrDir,audioNames,'.csv']);

%% spectrogram baseline-4:
    [spectr_base]=spectr_base_res04(x);
    writematrix(spectr_base.',[ftrDir,audioNames,'.csv']);
    disp(['the current audio No. is: ',num2str(au)])


end