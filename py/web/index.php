<?php
?>
<html lang="de">
<head>
    <title>Bootstrap Example</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
    <link rel="stylesheet" href="./css/main.css">
    <script type="text/javascript" src="/eel.js"></script>
    <script type="text/javascript">
        eel.expose(say_hello_js); // Expose this function to Python
        function say_hello_js(x) {
            console.log("Hello from " + x);
        }

        say_hello_js("Javascript World!");
        eel.say_hello_py("Javascript World!"); // Call a Python function
    </script>
</head>

<body>
<div class="maincol">
    <header>
        <h1 class="h1">Visual Symbol Evaluation</h1>
    </header>

    <main class="container-fluid maincol">
        <div class="row maincol">
            <div class="col-md-5 leftcol">
                <br>

                <div id="msg"></div>
                <form method="post" id="image-form">
                    <input type="file" name="img[]" class="file" accept="image/*">
                    <div class="input-group my-3">
                        <input type="text" class="form-control" disabled placeholder="Upload" id="file">
                        <div class="input-group-append">
                            <button type="button" class="browse btn btn-primary">Browse...</button>

                        </div>
                </form>
            </div>
            <br>
            <br>
            <div class="btn-group-lg mx-auto">
                <button type="button" class="btn btn-dark">Training</button>
                <button id="btn_testing" onclick="btn_testing()" type="button" class="btn btn-dark">Testing</button>
            </div>
            <br>
            <br>
            <p>Recognized values:</p>
            <textarea rows="2" class="w-100">
                Sample Output
            </textarea>
            <br>
            <br>
            <p>Verify:</p>

            <form>
                <div>
                    <input type="radio" id="choice1"
                           name="contact" value="True">
                    <label for="choice1">True</label>

                    <input type="radio" id="choice2"
                           name="contact" value="False">
                    <label for="choice2">False</label>
                    <div>
                        <button class="btn-dark" type="submit">Submit</button>
                    </div>
                </div>
            </form>
        </div>
        <div class="col-md-7">
            <img src="https://placehold.it/800x600" id="preview" class="img-fluid" alt="Responsive image">
        </div>
</div>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
<script type="text/javascript" src="./js/upload.js"></script>
<script type="text/javascript" src="./js/eel_test.js"></script>
</body>
</html>
