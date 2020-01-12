

function print_result(values) {
	document.getElementById("textpred").innerText = values;
}

function btn_reset() {
    pfusch= 1;
    eel.delimgs();
}


function btn_testing() {
	download("helloWorld.png");
	//eel.numberofcontours("C:\\Users\\%user\\Downloads\\helloWorld.png");
	//console.log("bis da her");
    eel.predict_image_with_existing_model("C:\\Users\\jakob\\Downloads\\helloWorld.png")(print_result);
    if (pfusch==1){
     pfusch = 0;
    btn_testing();
   
    }
}
