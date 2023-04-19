function iemocap_9946au_80dim_ftr_extract_ieee2021

agudir='/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/';
audir=dir([agudir,'allAu_SyTag_r/*.wav']);
au_names={audir.name}';
wav_sourcePath='/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/allAu_SyTag_r/';
ftrDir='/Users/xingfengli/Documents/RESEARCH/MusicRep/proposed_csvIEMOCAP_80/';

for au=6063:length(au_names)

    %loading audio file
    [x,fs]=audioread([wav_sourcePath,au_names{au}]);
    
 	[audio_in,audioNames,dirType]=eng_audioInfo_pre_loading(x);
	x=audio_in;
%% Proposed @2021
	[f0,loc] = pitch(x,fs,'Method','LHS','WindowLength',640,'OverlapLength',480);
	[fMIDI_V]=fo_embedding_interval_test_Ver2(f0);
	[musicTexture_vec,~]=eng_music_texture_tag_test_plus_wordTag2(audioNames,fs,f0,loc,fMIDI_V,0,dirType,0);
            
	[~,ex]=energyop(x,0);
	[TEO,S_teoInfo]=energy_short_timeTEO(ex,f0,fMIDI_V);
            
	[musicV2]=music_rep_emb_test(x,fs);
	[musicV2_melody]=music_rep_emb_melody_test(x,fs);
	[musicV2_Interval]=music_rep_emb_interval_test(musicV2,musicV2_melody);

	[S_ratiosV]=interval_ratios_embedding(musicTexture_vec,fMIDI_V);
	[primeFactorVec_and_StepVec_sel]=list_of_music_interval_demo(f0);

	[Mv_out,Dur]=eng_music_texture_frame_phoneme_syllable_word_embeds_new(audioNames,fs,f0,loc,fMIDI_V,0,dirType,0);
	[musicTexture_Wd,~]=eng_music_texture_word_level(audioNames,fs,f0,loc,fMIDI_V,0,dirType);%3-2-80

% 	featureVector=[ musicV2_Interval fMIDI_V ...
%         TEO S_teoInfo ...
%         S_ratiosV(:,[ 1 2 3 7 8 9 10 ])...
%         primeFactorVec_and_StepVec_sel(:,[1 3 4 5 6 7 8 9 13])...
%         Mv_out(:,[1 2 3 4 5 6 11 12 17 18])...
%         Dur(:,1)];
	featureVector=[ musicV2_Interval fMIDI_V ...
        TEO S_teoInfo ...
        S_ratiosV(:,[ 1 2 3 7 8 9 10 ])...
        primeFactorVec_and_StepVec_sel(:,[1 3 4 5 6 7 8 9 13])...
        Mv_out(:,[1 2]) musicTexture_vec(:,1:2) musicTexture_Wd(:,1:2)...
        Mv_out(:,[3 4 5 6 11 12 17 18])...
        Dur(:,1)];

    writematrix(featureVector.',[ftrDir,audioNames,'.csv']);
    disp(['the current audio No. is: ',num2str(au)])


end