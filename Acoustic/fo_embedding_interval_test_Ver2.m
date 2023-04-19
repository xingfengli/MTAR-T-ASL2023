function  [fMIDI_V]=fo_embedding_interval_test_Ver2(f0raw)
    %color_arrary:midi note v.s. octave    
    color_array=zeros(12,12);
    for r=1:12
        for c=1:12
            color_array(r,c)=(r-1)+12*(c-1);
        end  
    end
    
    %midi_map
    semitoneOffsets = [0, -0.25, -1/3, -0.5, -2/3, -0.75];
    shiftFB=0;
    k=shiftFB+1;
    midi = (0:143);                     % midi notes
    midi_freq = 2.^((midi-69+semitoneOffsets(k))/12)*440; 
   
    %fo->midi mappings
    fo_len=length(f0raw);  
    midiTmp=[];%1st emb
    for l= 1:fo_len
        
        fo_info=f0raw(l);
        [~,Id]=min(abs(midi_freq'-fo_info));
        midi_Id=Id-1;
        [m_row,m_col]=find(color_array==midi_Id);
        %corresponding [row] and [col]
        if fo_info==0
            m_row=0;
            m_col=0;
            midi_Id=0;
        end

        midiTmp=[midiTmp;[midi_Id m_row m_col]];
    end

    up_tag=[];%2nd emb
    for lnth=1:size(midiTmp,1)
        if lnth~=size(midiTmp,1)
            if midiTmp(lnth+1)-midiTmp(lnth)>0
                up_tag=[up_tag;1];
            else
                if midiTmp(lnth+1)-midiTmp(lnth)<0
                    up_tag=[up_tag;-1];
                else
                    up_tag=[up_tag;0];
                end
            end

        else
            up_tag=[up_tag;0];
        end
    end
    
    midiNotesC=midiTmp(:,1);
    quantile_tag=zeros(length(midiNotesC),1);
    y=quantile(midiNotesC(find(midiNotesC~=0)),[.025 .25 .50 .75 .975]);
    
    for x=1:length(midiNotesC)
        if midiNotesC(x)==0
            quantile_tag(x)=0;
        elseif midiNotesC(x)<=y(1)%0.025
            quantile_tag(x)=1;
        elseif midiNotesC(x)>y(1)&&midiNotesC(x)<=y(2)%0.025-0.25
            quantile_tag(x)=2;
        elseif midiNotesC(x)>y(2)&&midiNotesC(x)<=y(3)%0.25-0.5
            quantile_tag(x)=3;
        elseif midiNotesC(x)>y(3)&&midiNotesC(x)<=y(4)%0.5-0.75
            quantile_tag(x)=4;
        elseif midiNotesC(x)>y(4)&&midiNotesC(x)<=y(5)%0.75-0.95
            quantile_tag(x)=5;
        elseif midiNotesC(x)>y(5) %0.95-1
            quantile_tag(x)=6;
        end      
    end

   fMIDI=[f0raw midiTmp quantile_tag up_tag];
   midi_RLE_A=midiTmp(:,1);
%    Umidi=unique(midi_RLE_A);
%    midi_RLE_M=zeros(length(midi_RLE_A),2);
%    
%    for md=1:length(Umidi)
%       posM=find(midi_RLE_A==Umidi(md));
%       midi_RLE_M(posM,1)=length(posM);
%       for rmd=1:length(posM)
%           midi_RLE_M(posM(rmd),2)=rmd/length(posM);
%       end
%        
%    end
   
   
   midi_RLE_M_plus=zeros(length(midi_RLE_A),1);
   midi_RLE_M_plus(:,1)=midi_RLE_A(:,1)/144;
   
%    fMIDI_V=[fMIDI midi_RLE_M midi_RLE_M_plus];
   fMIDI_V=[fMIDI midi_RLE_M_plus];
%    id_mod=[1 2 3 4 ...
%        9 7 8 ...
%        6 5];
%    fMIDI_V=fMIDI_V(:,id_mod);
   
%01:f0 (Hz)
%02:midi Note Number
%03:midi Note(row)
%04:midi Octave(col.)
%05:quantile (Q1,Q3,etc.)
%06:up_tag(midi_after > midi_before)
%07:interval_tag: mod(abs(diff),12)
%08:music_order of interval
%09:music_type of interval

%10:the number of times that a certain midiNote (M) occurs in an utterance (N)
%11:the m_th times that M occurs in N, m_th/N

%12:the length of frames in an utterance (L)
%13:ratio between the number of the currencet frame (cl) and L: cl/L
%14:the relative position of 144 midiNotes 

%15:the number of times that a certain interval (I) occurs in an utterance (NI)
%16:the m_th times that I occurs in NI, m_th/NI
end

    