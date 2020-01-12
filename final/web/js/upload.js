
$(document).on("click", ".browse", function() {
  var file = $(this).parents().find(".file");
  // console.log(file);
 // window.alert(file);
  file.trigger("click");
});

$('input[type="file"]').change(function(e) {
  var fileName = e.target.files[0].name;
    console.log("2" + fileName);
  //window.alert(fileName);
  $("#file").val(fileName);

  var reader = new FileReader();
  reader.onload = function(e) {
    // get loaded data and render thumbnail.
    document.getElementById("preview").src = e.target.result;
  };
  // read the image file as a data URL.
 // document.getElementById("preview").value = e.target.result.toDataURL("image/png");
  document.getElementById("preview").value = reader.readAsDataURL(this.files[0]);
});