/*eel.expose(my_javascript_function);
function my_javascript_function(a, b) {
    console.log(a + b)
}
*/
function callback(){
    console.log("Callback");
}


/*async function btn_testing() {
    document.getElementById("btn_testing").style.color = "green";
    let val = JSON.stringify(eel.my_python_method(2));
    console.log("val JSON " + val);

	let arr = await JSON.stringify(eel.predict_image_with_existing_model(5));
	console.log("arr JSON" + arr);
	document.getElementById("textpred").innerHTML = arr;
    if(val < 0) {
        document.getElementById("btn_testing").style.color = "red";
    }else {
        document.getElementById("btn_testing").style.color = "blue";
    }
}*/

function print_result(values) {
	document.getElementById("textpred").innerText = values;
}

function btn_testing() {
	eel.delimgs()
	download("data:"+ image, "helloWorld.png");
	//eel.numberofcontours("C:\\Users\\%user\\Downloads\\helloWorld.png");
	//console.log("bis da her");
    eel.predict_image_with_existing_model("C:\\Users\\jakob\\Downloads\\helloWorld.png")(print_result);
}
