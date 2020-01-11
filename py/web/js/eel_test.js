/*eel.expose(my_javascript_function);
function my_javascript_function(a, b) {
    console.log(a + b)
}
*/
async function btn_testing() {
    document.getElementById("btn_testing").style.color = "green";
    let val = eel.my_python_method(2);

	let arr = await eel.predict_image_with_existing_model(5);
	console.log(arr);
	document.getElementById("textpred").innerHTML = arr;
    if(val < 0) {
        document.getElementById("btn_testing").style.color = "red";
    }else {
        document.getElementById("btn_testing").style.color = "blue";
    }
}
