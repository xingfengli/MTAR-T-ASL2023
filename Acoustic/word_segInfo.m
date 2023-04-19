function [word_tag_Out]=word_segInfo(a01_trans_words,frm_trans_syllable)
% a01_trans_words=[{'der'} ;{'lappen'};{'liegt'};{'auf'};{'dem'};{'eisschrank'}];
% frm_trans_syllable=[{'der'};{'la'};{'pen'};{'liegt'};{'auf'};{'dem'};{'eis'};{'schrank'}];

% a01_trans_words=[{'heute'} ;{'abend'};{'konnte'};{'ich'};{'es'};{'ihm'};{'sagen'}];
% frm_trans_syllable=[{'heu'};{'te'};{'('};{'a'};{'bend'};{'('};{'konnt'};{'te'};{'ich'};{'es'};{'ihm'};{'sa'};{'gen'}];

%tagW, row: id of syllables; col: id of words
tagW=zeros(size(frm_trans_syllable,1),size(a01_trans_words,1));
for f=1:size(frm_trans_syllable,1)
    for w=1:size(a01_trans_words,1)
        if contains(a01_trans_words{w},frm_trans_syllable{f})
            tagW(f,w)=1;
        end
    end
end
%tagW_len, mapping a syllable to words of possible
tagW_len=[];
for t=1:size(tagW,1)
    tagW_len=[tagW_len;length(find(tagW(t,:)==1))];
end
% word_tag_Out: wordsTag for each syllable
word_tag_Out=[];
if length(find(tagW_len==1))==length(tagW_len)
    %each syllable corresponds to an unique word.
    for to=1:size(tagW,1)
        pos=find(tagW(to,:)==1);
        word_tag_Out=[word_tag_Out;pos];
    end
else
    %one specific syllable exists in more than one word.
    if isempty(find(tagW_len==0))
        %locating the position of specific syllable
        pid=find(tagW_len~=1);
        %if pid==1, then it belongs to the 1st word.
        if pid==1
            tagW(1,:)=0;
            tagW(1,1)=1;
        else
            for p=1:length(pid)
                if pid(p)==1
                    tagW(1,:)=0;
                    tagW(1,1)=1;
                else
                    if pid(p)==size(tagW,1)
                        bfr_spec_aft=[tagW(pid(p)-1,:);tagW(pid(p),:)];
                    else
                        bfr_spec_aft=[tagW(pid(p)-1,:);tagW(pid(p),:);tagW(pid(p)+1,:)];
                    end
                    pid_spec=find(tagW(pid(p),:)==1);
                    bfr_spec_aft2=bfr_spec_aft(:,pid_spec);
                    for p2=1:size(bfr_spec_aft2,2)
                        pos=find(bfr_spec_aft2(:,p2)==1);
                        if length(pos)>1
                            tagW(pid(p),:)=0;
                            tagW(pid(p),pid_spec(p2))=1;
                        end
                    end
                end
            end
        end
        % word_tag_Out: wordsTag for each syllable
        for to=1:size(tagW,1)
            pos=find(tagW(to,:)==1);
            word_tag_Out=[word_tag_Out;pos(end)];
        end
        
    else
        %there exists unrecognized syllable '(', omit it and restore in
        %frm_trans_syllable2.
        frm_trans_syllable2=[];
        sil_pos=[];
        for fts=1:size(frm_trans_syllable,1)
            if ~strcmp(frm_trans_syllable{fts},'(')
                frm_trans_syllable2=[frm_trans_syllable2;frm_trans_syllable(fts)];
            else
               sil_pos=[sil_pos;fts]; 
            end
        end
        %tagW, row: id of syllables; col: id of words
        tagW=zeros(size(frm_trans_syllable2,1),size(a01_trans_words,1));
        for f=1:size(frm_trans_syllable2,1)
            for w=1:size(a01_trans_words,1)
                if contains(a01_trans_words{w},frm_trans_syllable2{f})
                    tagW(f,w)=1;
                end
            end
        end
        %tagW_len, mapping a syllable to words of possible
        tagW_len=[];
        for t=1:size(tagW,1)
            tagW_len=[tagW_len;length(find(tagW(t,:)==1))];
        end
        %pid:%one specific syllable exists in more than one word.
        pid=find(tagW_len~=1);
        for p=1:length(pid)
            %creat a word by connecting two syllables of the current and
            %previous ones
            if pid(p)==1
                new_str=frm_trans_syllable2{pid(p)};
            else
                new_str=[frm_trans_syllable2{pid(p)-1},frm_trans_syllable2{pid(p)}];
            end
            %whether this new word exists in current ones
            tmpW=0;
            for w=1:size(a01_trans_words,1)
                if contains(a01_trans_words{w},new_str)
                    tmpW=w;
                end
            end
            %if, yes. tag as existed ones
            if tmpW
                tagW(pid(p),:)=0;
                tagW(pid(p),tmpW)=1;
            else
                tagW(pid(p),:)=0;
                idPre=find(tagW(pid(p)-1,:)==1);
                tagW(pid(p),idPre+1)=1;
            end
        end
        % word_tag_Out: wordsTag for each syllable
        for to=1:size(tagW,1)
            pos=find(tagW(to,:)==1);
            word_tag_Out=[word_tag_Out;pos];
        end
        %modify the word_tag_Out by inserting '('
        if length(sil_pos)
            WTO2=[];
            for s=1:length(sil_pos)
                if s==1
                    word_tag_Out_tmp=word_tag_Out;
                else
                    word_tag_Out_tmp=word_tag_Out2;
                end
                word_tag_Out2=zeros(length(word_tag_Out)+s,1);
                for a=1:size(word_tag_Out_tmp)
                    if a<sil_pos(s)
                        word_tag_Out2(a,1)=word_tag_Out_tmp(a,1);
                    else
                        if a==sil_pos(s)
                            word_tag_Out2(a,1)=word_tag_Out_tmp(a,1);
                            word_tag_Out2(a+1:end,1)=word_tag_Out_tmp(a:end,1)+1;
                            break;
                        end
                    end
                end
                WTO2=word_tag_Out2;
            end
            word_tag_Out=WTO2;
        end
    end  
end
end