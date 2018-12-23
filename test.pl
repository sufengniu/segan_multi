opendir(dir,"../dataset_timit_gan/test0db")||die;
@file=readdir(dir);
close(dir);

for($i=0;$i<@file;$i++){
	if($file[$i]=~/[a-z]|[A-Z]/){
		system("./clean_wav.sh ../dataset_timit_gan/test0db/$file[$i] test-gen/");
	}
}
