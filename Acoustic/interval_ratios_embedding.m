function [S_ratiosV]=interval_ratios_embedding(musicTexture_vec,fMIDI_V)

midiTmp=fMIDI_V(:,2);
musicTpOd=[...
    [1 0];...
    [2 -1];...
    [2 1];...
    [3 -1];...
    [3 1];...
    [4 0];...
    [5 -2];...
    [5 0];...
    [6 -1];...
    [6 1];...
    [7 -1];...
    [7 1];...
    [8 0];...
    ];
diff=[];
odTp=[];
for lnth=1:length(midiTmp)
     if lnth~=size(midiTmp,1)
         interval=mod(abs(midiTmp(lnth+1)-midiTmp(lnth)),12);
         if interval==0&&midiTmp(lnth+1)-midiTmp(lnth)~=0
             interval=12;
         end
         diff=[diff;interval];
         odTp=[odTp;musicTpOd(interval+1,:)];
     else
         interval=0;
         diff=[diff;interval];
         odTp=[odTp;musicTpOd(interval+1,:)];
     end
end

syInfo=musicTexture_vec(:,5).*musicTexture_vec(:,6);

upInfo=fMIDI_V(:,6);
%1:up -1:down
minMaj=odTp(:,2);
%-1:minor 1:Maj 0:perfect
Usixteenths=unique(syInfo);
S_ratios=zeros(size(fMIDI_V,1),3);
% SyTEO_Info=zeros(size(fMIDI_V,1),2);
for u=1:length(Usixteenths)
    sil_tag0=find(syInfo==Usixteenths(u));
    upOccurs=upInfo(sil_tag0);
    if isempty(find(upOccurs==-1))
        S_ratios(sil_tag0,1)=0;
    else
        S_ratios(sil_tag0,1)=length(find(upOccurs==-1))/(length(find(upOccurs==-1))+length(find(upOccurs==1)));
    end
    
    minMajOccurs=minMaj(sil_tag0);
    if isempty(find(minMajOccurs==-1))
        S_ratios(sil_tag0,2)=0;
        S_ratios(sil_tag0,3)=length(find(minMajOccurs==0))/length(sil_tag0);
    else
        S_ratios(sil_tag0,2)=length(find(minMajOccurs==-1))/(length(find(minMajOccurs==-1))+length(find(minMajOccurs==1)));
        S_ratios(sil_tag0,3)=length(find(minMajOccurs==0))/length(sil_tag0);
    end 
    
%     SyTEO=TEO(sil_tag0);
%     avgInfo=mean(SyTEO);
%     stdInfo=std(SyTEO);
%     SyTEO_Info(sil_tag0,1)=avgInfo;
%     SyTEO_Info(sil_tag0,2)=stdInfo;
    
end

Udiff=unique(diff);
RLEs=zeros(length(diff),2);
%D=[];
for d=1:length(Udiff)
    pIdx=find( diff==Udiff(d) );
    %D=[D;[d length(pIdx)]]
    RLEs(pIdx,1)=length(pIdx);
    %RLEs(pIdx,3)=abs((length(pIdx)/length(diff))*log2(length(pIdx)/length(diff)));
    for p=1:length(pIdx)
        RLEs(pIdx(p),2)=p/length(pIdx);
    end
end

Umidi=unique(midiTmp);
RLEs2=zeros(length(midiTmp),2);
for d2=1:length(Umidi)
    pIdx2=find( midiTmp==Umidi(d2) );
    %D=[D;[d length(pIdx2)]]
    RLEs2(pIdx2,1)=length(pIdx2);
    %RLEs(pIdx2,3)=abs((length(pIdx2)/length(diff))*log2(length(pIdx2)/length(diff)));
    for p2=1:length(pIdx2)
        RLEs2(pIdx2(p2),2)=p2/length(pIdx2);
    end
end





S_ratiosV=[ diff odTp S_ratios RLEs RLEs2 ];


end