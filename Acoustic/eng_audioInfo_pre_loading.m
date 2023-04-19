function[audio_in,audioNames,dirType]=eng_audioInfo_pre_loading(x)
audio_in=x(1:end-1);
num_Of_dir=2;%1-ori 2-agu
dirInfo=round( 0.5*num_Of_dir );
dirType=dirInfo;

% if dirInfo==1
    %tempdir='/private/var/folders/vr/wg88wmg95hxdyv2xy8dk20hh0000gn/T/';
% 	audir=dir(['/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/iemocap2021_mod2/','*.wav']);
% 	au_names={audir.name}';
[data,~,raw]=xlsread(['/Users/xingfengli/Documents/RESEARCH/MusicRep/iemocap/IEMOCAP/sound/','IDNames9946.xlsx']);
wavStr=raw(2:end,2);
au=round(x(end)*length(data));
%  	au_str=int2str(au);
%  	%au_str = sprintf('%d',round(au));
% 	switch length(au_str)
%         case 1
%             au_str=['00',au_str];
%         case 2
%             au_str=['0',au_str];
%         case 3
%             au_str=au_str;
%     end
% 	audioNames=au_names{str2num(au_str)};
    audioNames=wavStr{au};
% else
% 	agudir='/Users/xingfengli/Documents/RESEARCH/MusicRep/';
% 	audir=dir([agudir,'augmentedData_mod2/*.wav']);
% 	au_names={audir.name}';
%     au=round(x(end-1)*length(au_names));
%     audioNames=au_names{au};
% end



end