classdef audioFeatureExtractor_MusicRepLi_Eng < handle & ...
        matlab.mixin.CustomDisplay & ...
        matlab.mixin.SetGet & ...
        matlab.mixin.Copyable
    
    methods
        
        function featureVector=extract(obj,x)
            
            fs=16000;
            [audio_in,audioNames,dirType]=eng_audioInfo_pre_loading(x);
            x=audio_in;
%% Proposed @2021
            [f0,loc] = pitch(x,fs,'Method','LHS','WindowLength',640,'OverlapLength',480);
            [fMIDI_V]=fo_embedding_interval_test_Ver2(f0);
            [musicTexture_vec,~]=eng_music_texture_tag_test_plus_wordTag2(audioNames,fs,f0,loc,fMIDI_V,0,dirType,0);
            %[bfr_Interval,aft_Interval,bfr_IntvlMnzHt,aft_IntvlMnzHt,RLE_Notes]=Note_intervalTest(fMIDI_V,musicTexture_vec);
            
            [~,ex]=energyop(x,0);
            [TEO,S_teoInfo]=energy_short_timeTEO(ex,f0,fMIDI_V);
            %[EKSS]= TEO_ex_spectral_Info_cal(ex,fs,0);
            
            [musicV2]=music_rep_emb_test(x,fs);
            [musicV2_melody]=music_rep_emb_melody_test(x,fs);
            [musicV2_Interval]=music_rep_emb_interval_test(musicV2,musicV2_melody);

            [S_ratiosV]=interval_ratios_embedding(musicTexture_vec,fMIDI_V);

            [primeFactorVec_and_StepVec_sel]=list_of_music_interval_demo(f0);

            [Mv_out,Dur]=eng_music_texture_frame_phoneme_syllable_word_embeds_new(audioNames,fs,f0,loc,fMIDI_V,0,dirType,0);

            %[S_ratiosPh]=interval_ratios_embedding_plus_phoneme_wordLevel(Mv_out(:,1:6),fMIDI_V,TEO);
            %[S_ratiosSy]=interval_ratios_embedding_plus_phoneme_wordLevel(Mv_out(:,7:12),fMIDI_V,TEO);
            %[S_ratiosWd]=interval_ratios_embedding_plus_phoneme_wordLevel(Mv_out(:,13:18),fMIDI_V,TEO);

            
            %[OpS_ftrs]=openSmile_eGeMAPS_globalInfo_loading(audioNames,f0);
            %[musicTexture_Ph,~]=eng_music_texture_phoneme_level(audioNames,fs,f0,loc,fMIDI_V,0,dirType);
            [musicTexture_Wd,~]=eng_music_texture_word_level(audioNames,fs,f0,loc,fMIDI_V,0,dirType);
    
%             featureVector=[ musicV2_Interval fMIDI_V...
%                 musicTexture_vec ...
%                 bfr_Interval...
%                 aft_Interval...
%                 bfr_IntvlMnzHt...
%                 aft_IntvlMnzHt ...
%                 RLE_Notes(:,1:2) ...
%                 TEO S_teoInfo ...
%                 S_ratiosV];
%             if size(featureVector,1)<20
%                 featureVector=[featureVector;zeros(20-size(featureVector,1),68)];
%             end
%             featureVector=[ musicV2_Interval fMIDI_V(:,[1 2 3 4 6])  ...
%                 musicTexture_vec ...
%                 TEO S_teoInfo ...
%                 S_ratiosV(:,[ 1 2 3 4 5 7 8 9 10 ])...
%                 primeFactorVec_and_StepVec_sel(:,[1 3 4 5 6 7 8 9 13])];
	featureVector=[ musicV2_Interval fMIDI_V ...
        TEO S_teoInfo ...
        S_ratiosV(:,[ 1 2 3 7 8 9 10 ])...
        primeFactorVec_and_StepVec_sel(:,[1 3 4 5 6 7 8 9 13])...
        Mv_out(:,[1 2]) musicTexture_vec(:,1:2) musicTexture_Wd(:,1:2)...
        Mv_out(:,[3 4 5 6 11 12 17 18])...
        Dur(:,1)];

% %% melSpectrogeam baseline-1:
%             [mel_base]=mel_base_res01(x,fs);
%             featureVector=mel_base;

% %% mfcc baseline-2:
%             [mfcc_base]=mfcc_base_res02(x,fs);
%             featureVector=mfcc_base;

% %% VGGish baseline-3:
%             [vggish_base]=vggish_base_res03(x,fs);
%             featureVector=vggish_base;

% %% spectrogram baseline-4:
%             [spectr_base]=spectr_base_res04(x);
%             featureVector=spectr_base;
             
        end
          
    end
    
end