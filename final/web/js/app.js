// Set constraints for the video stream
var constraints = { video: { facingMode: "user" }, audio: false };
var track = null;
var image = null;

// Define constants
const cameraView = document.querySelector("#camera--view"),
    cameraOutput = document.querySelector("#preview"),
    cameraSensor = document.querySelector("#camera--sensor"),
    cameraTrigger = document.querySelector("#camera--trigger");

// Access the device camera and stream to cameraView
function cameraStart() {
    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function(stream) {
            track = stream.getTracks()[0];
            cameraView.srcObject = stream;
        })
        .catch(function(error) {
            console.error("Oops. Something is broken.", error);
        });
} 

// Take a picture when cameraTrigger is tapped
cameraTrigger.onclick = function() {
    cameraSensor.width = cameraView.videoWidth;
    cameraSensor.height = cameraView.videoHeight;
    cameraSensor.getContext("2d").drawImage(cameraView, 0, 0);



	var cameraValue = cameraSensor.toDataURL("image/png");
    cameraOutput.src = cameraValue;
    image = cameraValue;
	//console.log(cameraValue)
	//var decrypted_cameraValue = decodeURIComponent(cameraValue);
	
   // document.getElementById("preview").src = cameraSensor.toDataURL("image/png");
   // cameraOutput.classList.add("taken");
    // track.stop();
};

function download(dataurl, filename) {
  var a = document.createElement("a");
  a.href = dataurl;
  a.setAttribute("download", filename);
  a.click();
}

function printresult(res){
	console.log(res);
}

// Start the video stream when the window loads
window.addEventListener("load", cameraStart, false);