var video = document.getElementById('video');

if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
	navigator.mediaDevices.getUserMedia({video: true}).then(function(stream){
		video.srcObject = stream;
		video.play();
	});
}