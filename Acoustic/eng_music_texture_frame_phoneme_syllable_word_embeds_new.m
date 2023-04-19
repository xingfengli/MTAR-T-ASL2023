function [Mv_out,Dur]=eng_music_texture_frame_phoneme_syllable_word_embeds_new(audioNames,fs,f0,loc,fMIDI_V,wordInfoOut_flag,dirType,seg_flag)

[musicTexture_Ph,Fr_Ph_dur]=eng_music_texture_phoneme_level(audioNames,fs,f0,loc,fMIDI_V,wordInfoOut_flag,dirType);
phNo=round(musicTexture_Ph(:,5).*musicTexture_Ph(:,6));

[musicTexture_Sy,Sy_dur]=eng_music_texture_tag_test_plus_wordTag2(audioNames,fs,f0,loc,fMIDI_V,wordInfoOut_flag,dirType,seg_flag);
syNo=round(musicTexture_Sy(:,5).*musicTexture_Sy(:,6));

Usy=unique(syNo);
syllable_pos_sturcture_array=zeros(size(musicTexture_Sy,1),6);
for s=1:length(Usy)
syllable_info=Usy(s);
pos=find(syNo==syllable_info);
uPh=unique(phNo(pos));
for p=1:length(uPh)%num of phonemes
posP=find(phNo(pos)==uPh(p));
syllable_pos_sturcture_array(pos(posP),1)=length(uPh);
syllable_pos_sturcture_array(pos(posP),2)=p/length(uPh);
syllable_pos_sturcture_array(pos(posP),3)=max(phNo);
syllable_pos_sturcture_array(pos(posP),4)=uPh(p)/max(phNo);
end
syllable_pos_sturcture_array(pos,5)=length(Usy);
syllable_pos_sturcture_array(pos,6)=Usy(s)/length(Usy);

end

[musicTexture_Wd,wd_dur]=eng_music_texture_word_level(audioNames,fs,f0,loc,fMIDI_V,wordInfoOut_flag,dirType);
wdNo=round(musicTexture_Wd(:,5).*musicTexture_Wd(:,6));

Uwd=unique(wdNo);
word_pos_sturcture_array=zeros(size(musicTexture_Wd,1),6);
for w=1:length(Uwd)
word_info=Uwd(w);
posw=find(wdNo==word_info);
uSy=unique(syNo(posw));
for p=1:length(uSy)%num of phonemes
posS=find(syNo(posw)==uSy(p));
word_pos_sturcture_array(posw(posS),1)=length(uSy);
word_pos_sturcture_array(posw(posS),2)=p/length(uSy);
word_pos_sturcture_array(posw(posS),3)=max(syNo);
word_pos_sturcture_array(posw(posS),4)=uSy(p)/max(syNo);
end
word_pos_sturcture_array(posw,5)=length(Uwd);
word_pos_sturcture_array(posw,6)=Uwd(w)/length(Uwd);

end

Mv_out=[musicTexture_Ph...
syllable_pos_sturcture_array...
word_pos_sturcture_array];
Dur=[Fr_Ph_dur...
Sy_dur...
wd_dur];













end