function [musicTexture_vec,syllable_duration]=eng_music_texture_tag_test_plus_wordTag2(audioNames,fs,f0,loc,fMIDI_V,wordInfoOut_flag,dirType,seg_flag)
%loading audio information
% [x,fs]=audioread('03a01Wa.wav');
% 
% %extract fundamental frequency
% [f0,loc] = pitch(x,fs,'Method','LHS','WindowLength',640,'OverlapLength',480);
% tempdir='/private/var/folders/vr/wg88wmg95hxdyv2xy8dk20hh0000gn/T/';
% downloadFolder = tempdir;
% datasetFolder = fullfile(downloadFolder,"Emo-DB");



% % % ads = audioDatastore(fullfile(datasetFolder,"wav"));
% % % 
% % % audio_store=readall(ads);
% % % % if find(cellfun(@(x) isequal(x,audioIn),audio_store))
% % % [pos_r,~]=find(cellfun(@(x) isequal(x,audioIn),audio_store));
% % % au_name_path=ads.Files{pos_r};
% % % au_name=ads.Files{pos_r}(max(find(ismember(au_name_path,'/')==1))+1:max(find(ismember(au_name_path,'.')==1))-1);
% % % % end
if dirType==1
    au_name=audioNames;
else
    au_name=audioNames(4:end-10);
end



% aug_tempdir='/Users/xingfengli/Documents/RESEARCH/';
% aug_downloadFolder = aug_tempdir;
% aug_datasetFolder = fullfile(aug_downloadFolder,"MusicRep");
% aug_ads = audioDatastore(fullfile(aug_datasetFolder,"augmentedData"));
% 
% aug_audio_store=readall(aug_ads);
% if find(cellfun(@(x) isequal(x,audioIn),aug_audio_store))
%     [aug_pos_r,~]=find(cellfun(@(x) isequal(x,audioIn),aug_audio_store));
%     aug_au_name_path=aug_ads.Files{aug_pos_r};
%     au_name=aug_ads.Files{aug_pos_r}(max(find(ismember(aug_au_name_path,'/')==1))+1:max(find(ismember(aug_au_name_path,'_')==1))-1);
% end

% transcript_id=au_name(3:5);
% switch transcript_id
%     case 'a01'
%         a01_trans_words=[{'der'} ;{'lappen'};{'liegt'};{'auf'};{'dem'};{'eisschrank'}];
%     case 'a02'
%         a01_trans_words=[{'das'} ;{'will'};{'sie'};{'am'};{'mittwoch'};{'abgeben'}];
%     case 'a04'
%         a01_trans_words=[{'heute'} ;{'abendmd'};{'könntek�nnte'};{'ich'};{'es'};{'ihm'};{'sagen'}];
%     case 'a05'
%         a01_trans_words=[{'das'} ;{'schwarze'};{'blatt'};{'papier'};{'befindet'};{'sich'};{'da'};{'oben'};{'bmneben'};{'dem'};{'holzstück'}];
%     case 'a07'
%         a01_trans_words=[{'in'} ;{'siebenbm'};{'stundendn'};{'wird'};{'es'};{'soweit'};{'sein'}];
%     case 'b01'
%         a01_trans_words=[{'was'} ;{'sind'};{'denn'};{'das'};{'für'};{'tüten'};{'die'};{'da'};{'unter'};{'dem'};{'tisch'};{'stehnstehenhn'}];
%     case 'b02'
%         a01_trans_words=[{'sie'} ;{'habenbmbn'};{'es'};{'grgerade'};{'hochgetragen'};{'und'};{'jetzt'};{'gehenhngehn'};{'sie'};{'wieder'};{'runter'}];
%     case 'b03'
%         a01_trans_words=[{'an'} ;{'den'};{'wochenenden'};{'bin'};{'ich'};{'jetzt'};{'immer'};{'nach'};{'Hauhause'};{'gefahrnrenfahren'};{'und'};{'habe'};{'Agagnesaknisagnes'};{'besucht'}];
%     case 'b09'
%         a01_trans_words=[{'ich'} ;{'will'};{'das'};{'eben'};{'wegbringengn'};{'und'};{'dann'};{'mit'};{'karl'};{'was'};{'trinken'};{'gehenhngehn'}];
%     case 'b10'
%         a01_trans_words=[{'die'} ;{'wird'};{'auf'};{'dem'};{'platz'};{'sein'};{'wo'};{'wir'};{'sie'};{'immer'};{'hinlegen'}];
% end

%reading time info. of syllables
%[time_info,str_info]

[SFrm,EFrm,SegSy]=textread(['/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/alignment/',au_name,'.syseg'],'%s %s %s');
% if  ~strcmp('Ses03M_impro03_M001',au_name)
%     [SFrm,EFrm,SegSy]=textread(['/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/syseg2021/',au_name,'.syseg'],'%s %s %s');
%     switch au_name
%         case 'Ses01F_script01_3_F010'
%             SFrm=[{'SFrm'};{'50'}];
%             EFrm=[{'EFrm'};{'100'}];
%             SegSy=[{'Syllale'};{'AHH'}];
%     end
% else
%     [SFrm,EFrm,SegSy]=textread(['/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/syseg2021/','Ses03M_impro03_M003.syseg'],'%s %s %s');
% end
% 
% if strcmp(au_name,'13a01Wb')
%     time_info=[{'0.073214'};...
%         {'0.208890'};...
%         {'0.414146'};...
%         {'0.587314'};...
%         {'0.907334'};...
%         {'1.095564'};...
%         {'1.329688'};...
%         {'1.718573'};...
%         {'2.293052'};...
%         ];
%     
%     str_info=[{'der'};...
%         {'la'};...
%         {'pen'};...
%         {'liegt'};...
%         {'auf'};...
%         {'dem'};...
%         {'eis'};...
%         {'schrank'};...
%         {'.'};...
%         ];
% end

% [pos_r,~]=find(cellfun(@(x) isequal(x, 'k�nnte'),str_info));
% [pos_r2,~]=find(cellfun(@(x) isequal(x, 'k�nn'),str_info));
% [pos_r3,~]=find(cellfun(@(x) isequal(x, 'A'),str_info));
% [pos_r4,~]=find(cellfun(@(x) isequal(x, 'st�ck'),str_info));
% [pos_r5,~]=find(cellfun(@(x) isequal(x, 'f�r'),str_info));
% [pos_r6,~]=find(cellfun(@(x) isequal(x, 't�'),str_info));

% if pos_r
%     str_info(pos_r)={'könnte'};
% end
% 
% if pos_r2
%     str_info(pos_r2)={'könn'};
% end
% 
% if pos_r3
%     str_info(pos_r3)={'a'};
% end
% 
% if pos_r4
%     str_info(pos_r4)={'stück'};
% end
% 
% if pos_r5
%     str_info(pos_r5)={'für'};
% end
% 
% if pos_r6
%     str_info(pos_r6)={'tü'};
% end


%location info: [start,end] time
% loc_boundary_info=[];
% str_boundary_info=[];
% for t=1:size(time_info,1)
%     if t==1
%         loc_boundary_info=[loc_boundary_info;[0 str2double(time_info{t,1})*fs]];%sil
%     else
%         loc_boundary_info=[loc_boundary_info;[str2double(time_info{t-1,1})*fs str2double(time_info{t,1})*fs]];%sil
%     end
% end
SFrm_num=[];
for s=2:length(SFrm)
    SFrm_num=[SFrm_num;str2double(SFrm{s})];
end
EFrm_num=[];
for e=2:length(EFrm)
    EFrm_num=[EFrm_num;str2double(EFrm{e})];
end

SEFrm_Rgl=[SFrm_num EFrm_num];
%for r=1:size(SEFrm_Rgl,1)-1
r=1;
rowNum=size(SEFrm_Rgl,1) ;
while r<rowNum
    if abs(SEFrm_Rgl(r,2)-SEFrm_Rgl(r+1,1))==1
        r=r+1;
        continue;
    else
        if abs(SEFrm_Rgl(r,2)-SEFrm_Rgl(r+1,1))<80
            SEFrm_Rgl(r,2)=SEFrm_Rgl(r+1,1)-1;
            r=r+1;
        else
            SEFrm_Rgl=[SEFrm_Rgl(1:r,:);...
                [SEFrm_Rgl(r,2)+1 SEFrm_Rgl(r+1,1)-1];SEFrm_Rgl(r+1:end,:)];
            SegSy=[SegSy(1:(r+1),1);{'sil'};SegSy(r+2:end,1)];
            rowNum=size(SEFrm_Rgl,1) ;
            r=r+1;
        end
    end
            
end 
%end
SFrm_num2=SEFrm_Rgl(:,1);
EFrm_num2=SEFrm_Rgl(:,2);

if seg_flag==1
start_end_syInfo=[ SFrm_num2 EFrm_num2 ];
start_end_syInfo2=start_end_syInfo-SFrm_num2(1);
loc_boundary_info=[(start_end_syInfo2(:,1)/100)*fs (start_end_syInfo2(:,2)/100)*fs];%[loc_boundary_info;[str2double(time_info{end,1})*fs loc(end)]];
str_boundary_info=SegSy(2:end);

else
    start_end_syInfo=[ [0 ((SFrm_num2(1)-1)/100)*fs];[(SFrm_num2/100)*fs (EFrm_num2/100)*fs];...
        [((EFrm_num2(end)+1)/100)*fs loc(end) ]];
    loc_boundary_info=start_end_syInfo;
    str_boundary_info=[{'sil'};SegSy(2:end);{'sil'}];
    
end

if wordInfoOut_flag
    [word_tag_Out]=word_segInfo(a01_trans_words,str_boundary_info(2:end-1));
    word_tagVector=zeros(length(loc),1);%no. tag of words
end
%counting and mapping the number of syllables for f0
texture_tag=zeros(length(loc),1);%no. tag of syllables
str_sil_tag=cell(length(loc),1);

for l=1:length(loc)
    for r=1:size(loc_boundary_info,1)
        if loc(l)>=loc_boundary_info(r,1)&&loc(l)<=loc_boundary_info(r,2)
            texture_tag(l)=r;
            str_sil_tag(l)=str_boundary_info(r);
        end
    end
if wordInfoOut_flag    
    if find(ismember(str_boundary_info(2:end-1),str_sil_tag{l})==1)
        index=find(ismember(str_boundary_info(2:end-1),str_sil_tag{l})==1);
        if length(index)>1
            if l==1
                tf_thre=1;
            else
                for k=1:size(str_sil_tag(1:l-1),1)
                    if ~ strcmp(str_sil_tag{k},str_sil_tag{l})
                        tf_thre=k;
                    end
                end
            end

            new_str=[str_sil_tag{tf_thre},str_sil_tag{l}];
            tmpW=0;
            for w=1:size(a01_trans_words,1)
                if contains(a01_trans_words{w},new_str)
                    tmpW=w;
                end
            end

            if tmpW
                word_tagVector(l)=word_tagVector(tf_thre);
                continue;
            else
                idPre=word_tagVector(tf_thre);
                index=idPre+1;
            end
        end
        word_tagVector(l)=word_tag_Out(index);
    end
end
end

if wordInfoOut_flag
    start_id=min(find(word_tagVector==1));
    end_id=max(find(word_tagVector==max(word_tagVector)));
    
    mod_A=[];
    for q=start_id:end_id-1
        var_mod=word_tagVector(q+1)-word_tagVector(q);
        if var_mod==0||var_mod==1
            continue;
        else
            mod_A=[mod_A;q];
        end
    end
    
    %word_tagVector
    if ~isempty(mod_A)&&length(mod_A)~=1
        word_tagVector(mod_A(1)+1:mod_A(1)+round(0.5*(mod_A(2)-mod_A(1))))=word_tagVector(mod_A(1));
        word_tagVector(mod_A(1)+round(0.5*(mod_A(2)-mod_A(1)))+1:mod_A(2))=word_tagVector(mod_A(2)+1);
    end

end

%embedding the duration information for each uttered syllable, and the
%relative position of a certain syllable in the utterance
uTag=unique(texture_tag);
dur_frm_info=zeros(length(f0),1);%duration and relative number of syllables
for u=1:length(uTag)
    if u==1||u==length(uTag)
        dur_frm_info(find(texture_tag==uTag(u)),1)=0;%silence: unvoiced
    else
        pos_tmp=find(texture_tag==uTag(u));
        if strcmp(str_sil_tag{pos_tmp(1)},'sil')||strcmp(str_sil_tag{pos_tmp(1)},'SIL')
            dur_frm_info(find(texture_tag==uTag(u)),1)=0;%silence: unvoiced
        else
            dur_frm_info(find(texture_tag==uTag(u)),1)=1;%voiced
        end
    end
% % %     %the relative pos of the syllable in the current sentence
% % %     dur_frm_info(find(texture_tag==uTag(u)),2)=uTag(u)/length(uTag);
% % %     %duration of the syllable
% % %     dur_frm_info(find(texture_tag==uTag(u)),3)=length(find(texture_tag==uTag(u)))*10;%10ms
% % %     %relative duration: duration_one_syllable/sum(duration_all_syllables)
% % %     dur_frm_info(find(texture_tag==uTag(u)),4)=(length(find(texture_tag==uTag(u)))*10)/(10*length(texture_tag));%10ms
end

% % % syllable_in_word_Info=zeros(size(word_tagVector,1),2);
% % % word_dur_info=zeros(size(word_tagVector,1),1);
% % % uwords=unique(word_tagVector);
% % % uwords=uwords(2:end);
% % % for uw=1:length(uwords)
% % %     posuw=find(word_tagVector==uwords(uw));
% % %     if uw==1
% % %         word_dur_info(1:posuw(1)-1,1)=10*length(1:posuw(1)-1);
% % %     end
% % %     word_dur_info(posuw,1)=10*length(posuw);
% % %     if uw==length(uwords)
% % %         word_dur_info(posuw(end)+1:end,1)=10*length([posuw(end)+1:size(word_dur_info,1)]);
% % %     end
% % %     syllableInfo=texture_tag(posuw,1);
% % %     sy_count=unique(syllableInfo);
% % %     syllable_in_word_Info(posuw,1)=length(sy_count);
% % %     for sc=1:length(sy_count)
% % %         syllable_in_word_Info(find(texture_tag==sy_count(sc)),2)=sc/ unique(syllable_in_word_Info(posuw,1));
% % %     end
% % % end
% % % 
% % % musicTexture_vec=[texture_tag dur_frm_info word_tagVector word_tagVector/max(word_tagVector)...
% % %     syllable_in_word_Info word_dur_info word_dur_info/(size(word_dur_info,1)*10) dur_frm_info(:,3)./word_dur_info];
%01:texture_tag, the id of syllables in an utterance(start syllable:sil, end syllable:'.')

%02:voiced_and_unvoiced,0-sil; 1-voiced
%03:rel. position of syllables, e.g. 10 syllables in 1 utterance, sil 1/10
%04:duration that the current syllable lasts. a~~~~(40ms)
%05:ratio between the duration of the current syllable and whole utterance

%06:the id of words in an utterance(start and end were non-word, sil)
%07:rel. position of words, e.g. 6 words in 1 utterance, der(1),=1/6

%08:the total number of syllables in a word, eisschrank=eis+schrank(2)
%09:the rel. pos. of a syllable in a certain word, eis 1/2, schrank 2/2

%10:the dur. that a word lasts in an utterance.
%11:the ratio between the dur. of a word and a whole utterance
%12:the ratio betwwen the dur. of a syllable and a word

Hcodes_utterance=fMIDI_V(:,2);

%midi_map
semitoneOffsets = [0, -0.25, -1/3, -0.5, -2/3, -0.75];
shiftFB=0;
k=shiftFB+1;
midi = (0:143);                     % midi notes
midi_freq = 2.^((midi-69+semitoneOffsets(k))/12)*440;  
%fo->midi mappings
f0raw=f0;
fo_len=length(f0raw);  
midiTmp=[];%1st emb
for l= 1:fo_len
    fo_info=f0raw(l);
    [~,Id]=min(abs(midi_freq'-fo_info));
    midi_Id=Id-1;
    %corresponding [row] and [col]
    if fo_info==0
        midi_Id=0;
    end
    midiTmp=[midiTmp;midi_Id];
end
    
frame_duration=zeros(size(texture_tag,1),1);
uMD=unique(midiTmp);
for um=1:length(uMD)
    posM=find(midiTmp==uMD(um));
    B = [0 find(diff(posM')~=1) length(posM')];
    for i = 1:length(B)-1
       frame_duration(posM((B(i)+1):B(i+1)),1)=length( posM((B(i)+1):B(i+1)) )*10;
    end
end

syllable_pos_sturcture_array=zeros(size(texture_tag,1),6);
uniqSyllable=unique(texture_tag);
syllable_duration=zeros(size(texture_tag,1),1);
% RLE_in_syllable=zeros(size(texture_tag,1),1);
HC_midi=zeros(length( Hcodes_utterance),3);

for us=1:length(uniqSyllable)
    syllable_info=uniqSyllable(us);
    syllable_pos_sturcture_array(find(texture_tag==syllable_info),1)=length(find(texture_tag==syllable_info));
    posS=find(texture_tag==syllable_info);
    for ls=1:length(posS)
        syllable_pos_sturcture_array(posS(ls),2)=ls/length(posS);
        syllable_pos_sturcture_array(posS(ls),4)=posS(ls)/size(texture_tag,1);
    end
    syllable_pos_sturcture_array(find(texture_tag==syllable_info),3)=size(texture_tag,1);
    syllable_pos_sturcture_array(find(texture_tag==syllable_info),5)=length(uniqSyllable);
    syllable_pos_sturcture_array(find(texture_tag==syllable_info),6)=syllable_info/length(uniqSyllable);
    
    syllable_duration(posS,1)=length(posS)*10;%10ms
%     uMDS=unique(midiTmp(posS));
%     for ds=1:length(uMDS)
%         RLE_in_syllable(posS(find(midiTmp(posS)==uMDS(ds))),1)=length(find(midiTmp(posS)==uMDS(ds)));
%     end

    spec_syllable_str=Hcodes_utterance(posS);
    Hcodes=spec_syllable_str;
    UHuffCodes=unique(Hcodes);
    HCfreq=zeros(length( Hcodes),1);
    UHC_Freq=[];
    for hc=1:length(UHuffCodes)
        h_tmp=find( Hcodes==UHuffCodes(hc) );
        HCfreq(h_tmp)= length(h_tmp)/length(Hcodes);
        UHC_Freq=[UHC_Freq;length(h_tmp)/length(Hcodes)];
    end
    %us
    UHC_Freq=[UHuffCodes UHC_Freq];
    if size(UHC_Freq,1)==1
        dict=[{[UHC_Freq(1,1)]} {[1]}];
    else
        [dict,avglen]  = huffmandict(UHC_Freq(:,1)',UHC_Freq(:,2)');
    end
    Dec=[];
    
    for dictId=1:size(dict,1)
        Dec=[Dec;[bin2dec(num2str(dict{dictId,2})) length(dict{dictId,2})]];
    end
    Dec=[UHC_Freq Dec];
    for decId=1:size(Dec,1)
        decTmp=find(Hcodes==Dec(decId,1));
        HC_midi(posS(decTmp),1)=Dec(decId,2);
        HC_midi(posS(decTmp),2)=Dec(decId,3);
        HC_midi(posS(decTmp),3)=Dec(decId,4);
    end
    
end

if wordInfoOut_flag
word_pos_structure_array=zeros(size(texture_tag,1),6);
uniqWord=unique(word_tagVector);
uniqWord=uniqWord(2:end);%omit sil: sil is not a word.

word_duration=zeros(size(texture_tag,1),1);
% RLE_in_word=zeros(size(texture_tag,1),1);
HC_midi_word=zeros(length( Hcodes_utterance),3);

for uw=1:length(uniqWord)
    word_info=uniqWord(uw);
    posW=find(word_tagVector==word_info);
    num_sy_inWord=unique(texture_tag(posW,1));
    word_pos_structure_array(posW,1)=length(num_sy_inWord);
    for ns=1:length(num_sy_inWord)
        word_pos_structure_array(find(texture_tag==num_sy_inWord(ns)),2)=ns/length(num_sy_inWord);
        ns_pos=find(unique(texture_tag(find(word_tagVector~=0),1))==num_sy_inWord(ns));
        word_pos_structure_array(find(texture_tag==num_sy_inWord(ns)),4)=ns_pos/length(unique(texture_tag(find(word_tagVector~=0),1)));
    end
    word_pos_structure_array(posW,3)=length(unique(texture_tag(find(word_tagVector~=0),1)));
    word_pos_structure_array(posW,5)=length(uniqWord);
    word_pos_structure_array(posW,6)=word_info/length(uniqWord); 
    
    word_duration(posW,1)=length(posW)*10;
%     uMDW=unique(midiTmp(posW));
%     for dw=1:length(uMDW)
%         RLE_in_word(posW(find(midiTmp(posW)==uMDW(dw))),1)=length(find(midiTmp(posW)==uMDW(dw)));
%     end
    spec_word_str=Hcodes_utterance(posW);
    Hcodes=spec_word_str;
    UHuffCodes=unique(Hcodes);
    HCfreq=zeros(length( Hcodes),1);
    UHC_Freq=[];
    for hc=1:length(UHuffCodes)
        h_tmp=find( Hcodes==UHuffCodes(hc) );
        HCfreq(h_tmp)= length(h_tmp)/length(Hcodes);
        UHC_Freq=[UHC_Freq;length(h_tmp)/length(Hcodes)];
    end
    
    UHC_Freq=[UHuffCodes UHC_Freq];
    if size(UHC_Freq,1)==1
        dict=[{[UHC_Freq(1,1)]} {[1]}];
    else
        [dict,avglen]  = huffmandict(UHC_Freq(:,1)',UHC_Freq(:,2)');
    end
    Dec=[];
    
    for dictId=1:size(dict,1)
        Dec=[Dec;[bin2dec(num2str(dict{dictId,2})) length(dict{dictId,2})]];
    end
    Dec=[UHC_Freq Dec];
    for decId=1:size(Dec,1)
        decTmp=find(Hcodes==Dec(decId,1));
        HC_midi_word(posW(decTmp),1)=Dec(decId,2);
        HC_midi_word(posW(decTmp),2)=Dec(decId,3);
        HC_midi_word(posW(decTmp),3)=Dec(decId,4);
    end
    
end

sil_tag0=find(word_tagVector==0);
if ~isempty(sil_tag0)
    B2 = [0 find(diff(sil_tag0')~=1) length(sil_tag0')];
    for i = 1:length(B2)-1
        posSW=sil_tag0((B2(i)+1):B2(i+1));
%     usil_MDW=unique(midiTmp(sil_tag0((B2(i)+1):B2(i+1))));
%     for sdw=1:length(usil_MDW)
%         RLE_in_word(posSW(find(midiTmp(sil_tag0((B2(i)+1):B2(i+1)))==usil_MDW(sdw))),1)=length(find(midiTmp(sil_tag0((B2(i)+1):B2(i+1)))==usil_MDW(sdw)));
%     end
        spec_word_str=Hcodes_utterance(posSW);
        Hcodes=spec_word_str;
        UHuffCodes=unique(Hcodes);
        HCfreq=zeros(length( Hcodes),1);
        UHC_Freq=[];
        for hc=1:length(UHuffCodes)
            h_tmp=find( Hcodes==UHuffCodes(hc) );
            HCfreq(h_tmp)= length(h_tmp)/length(Hcodes);
            UHC_Freq=[UHC_Freq;length(h_tmp)/length(Hcodes)];
        end
    
        UHC_Freq=[UHuffCodes UHC_Freq];
        if size(UHC_Freq,1)==1
            dict=[{[UHC_Freq(1,1)]} {[1]}];
        else
            [dict,avglen]  = huffmandict(UHC_Freq(:,1)',UHC_Freq(:,2)');
        end
        Dec=[];
        
        for dictId=1:size(dict,1)
            Dec=[Dec;[bin2dec(num2str(dict{dictId,2})) length(dict{dictId,2})]];
        end
        Dec=[UHC_Freq Dec];
    
        for decId=1:size(Dec,1)
            decTmp=find(Hcodes==Dec(decId,1));
            HC_midi_word(posSW(decTmp),1)=Dec(decId,2);
            HC_midi_word(posSW(decTmp),2)=Dec(decId,3);
            HC_midi_word(posSW(decTmp),3)=Dec(decId,4);
        end
    end
end

word_duration(find(word_duration==0),1)=syllable_duration(find(word_duration==0),1);
% utterance_duration=ones(size(texture_tag,1),1)*size(texture_tag,1)*10;
%ratioScale_duration=[frame_duration./syllable_duration syllable_duration./word_duration word_duration./utterance_duration];
end
if wordInfoOut_flag
% % %     musicTexture_vec=[dur_frm_info syllable_pos_sturcture_array word_pos_structure_array...
% % %         frame_duration syllable_duration word_duration HC_midi HC_midi_word];
    musicTexture_vec=[ syllable_pos_sturcture_array frame_duration...
         word_pos_structure_array];% syllable_duration word_duration 
else
%     musicTexture_vec=[dur_frm_info syllable_pos_sturcture_array...
%         frame_duration syllable_duration HC_midi];

%     musicTexture_vec=[syllable_pos_sturcture_array...
%         frame_duration];%dur_frm_info syllable_duration];
    musicTexture_vec=syllable_pos_sturcture_array;

end

end