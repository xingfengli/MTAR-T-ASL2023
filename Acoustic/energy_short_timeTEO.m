function [TEO,S_teoInfo]=energy_short_timeTEO(ex,f0,fMIDI_V)
win_len=640;
win_hop=480;
win_shift=win_len-win_hop;
frames=floor(length(ex)/win_shift);
TEO=zeros(frames,1);
for f=1:frames
    if (win_len+(f-1)*win_shift)>length(ex)
        posId=[((f-1)*win_shift+1):length(ex)];
    else
        posId=[((f-1)*win_shift+1):(win_len+(f-1)*win_shift)];
    end
    TEO(f,1)=sum(ex(posId));
end
id=length(TEO)-length(f0);
startId=id+1;
TEO=TEO(startId:end);

midiNote=fMIDI_V(:,2);
sixteenthNotes=midiNote;
Usixteenths=unique(sixteenthNotes);
S_teoInfo=zeros(length(TEO),2);

for u=1:length(Usixteenths)
    sil_tag0=find(sixteenthNotes==Usixteenths(u));
    B2 = [0 find(diff(sil_tag0')~=1) length(sil_tag0')];
    for i = 1:length(B2)-1
        pos=sil_tag0((B2(i)+1):B2(i+1));
        avgInfo=mean(TEO(pos));
        stdInfo=std(TEO(pos));
        S_teoInfo(pos,1)=avgInfo;
        S_teoInfo(pos,2)=stdInfo;

    end  
end
end