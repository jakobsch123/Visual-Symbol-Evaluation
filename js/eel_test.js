/*eel.expose(my_javascript_function);
function my_javascript_function(a, b) {
    console.log(a + b)
}
*/
async function btn_testing() {
    document.getElementById("btn_testing").style.color = "green";
    let val = await eel.my_python_method(2);

    if(val < 1) {
        document.getElementById("btn_testing").style.color = "red";
    }else {
        document.getElementById("btn_testing").style.color = "blue";
    }
}
