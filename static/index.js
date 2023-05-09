(() => {
  // The width and height of the captured photo. We will set the
  // width to the value defined here, but the height will be
  // calculated based on the aspect ratio of the input stream.

  // |streaming| indicates whether or not we're currently streaming
  // video from the camera. Obviously, we start at false.

  let streaming = false;

  // The various HTML elements we need to configure or control. These
  // will be set by the startup() function.

  let video = null;
  let photoCanvas = null;
  let photo = null;
  let takePhotoButton = null;
  let showDetectionCheckbox = null;
  let previewCanvas = null;
  let showDetection = false;
  let predictionsContainer = null;
  let predictionTemplate = null;
  let predictionPartTemplate = null;
  let boundingBoxes = [];
  let crop = { x: 0, y: 0, width: 300, height: 300 };

  function startup() {
    var myInput = document.getElementById('myFileInput');

    function sendPic(event) {
      if (event.target.files && event.target.files[0]) {
        const reader = new FileReader();
        reader.onload = async (e) => {
          const dataUrl = e.target.result;
          const base64Image = dataUrl.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
          classify(base64Image);
        }
        reader.readAsDataURL(event.target.files[0]);
      }
    }
    myInput.addEventListener('change', sendPic, false);

    video = document.getElementById("video");
    previewCanvas = document.getElementById("preview-canvas");
    previewContext = previewCanvas.getContext("2d");

    // When you set the size of the canvas using CSS, it only affects the displayed size
    // of the canvas and doesn't change the actual drawing buffer's width and height.
    // As a result, you might experience issues with scaling and distortion.
    // These should match the style on the HTML element
    previewCanvas.width = 400;
    previewCanvas.height = 400;

    photoCanvas = document.getElementById("photo-canvas");
    // match size of image needed for prediction
    photoCanvas.setAttribute("height", 224);
    photoCanvas.setAttribute("width", 224);

    photo = document.getElementById("photo");
    takePhotoButton = document.getElementById("take-photo");

    showDetectionCheckbox = document.getElementById("show-detection-checkbox");

    predictionsContainer = document.getElementById("predictions-container");
    predictionTemplate = document.getElementById("prediction-template")
    predictionPartTemplate = document.getElementById("prediction-part-template")

    function updatePreviewCanvas() {
      previewContext.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, previewCanvas.width, previewCanvas.height);
      boundingBoxes.forEach((box) => {
        previewContext.strokeStyle = box.valid ? 'red' : 'gray';
        previewContext.lineWidth = 2;

        // Detection is done on 244x244 image, scale to the 400x400 preview
        const widthScaleFactor = 400 / 224;
        const heightScaleFactor = 400 / 224;

        // Draw the rectangle
        previewContext.beginPath();
        previewContext.rect(box.x * widthScaleFactor, box.y * heightScaleFactor, box.w * widthScaleFactor, box.h * heightScaleFactor);
        previewContext.stroke();
      });
      requestAnimationFrame(updatePreviewCanvas);
    }

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        video.srcObject = stream;
        video.play();

        requestAnimationFrame(updatePreviewCanvas)
      })
      .catch((err) => {
        console.error(`An error occurred: ${err}`);
        document.getElementById("classes").innerText = err;
      });

    video.addEventListener(
      "canplay",
      (ev) => {
        if (!streaming) {
          // determine porttion of the video frame we want to fetch
          const zoom = 4;
          const size = Math.min(video.videoWidth, video.videoWidth) / zoom;
          crop = { x: (video.videoWidth - size) / 2, y: (video.videoHeight - size) / 2, width: size, height: size };
          streaming = true;
        }
      },
      false
    );

    showDetectionCheckbox.addEventListener(
      "click",
      (ev) => {
        showDetection = showDetectionCheckbox.checked;
        boundingBoxes = [];
        if (showDetection) {
          detectionLoop();
        }
      }
    )

    takePhotoButton.addEventListener(
      "click",
      (ev) => {
        takepicture();
        ev.preventDefault();
      },
      false
    );

    clearphoto();
  }

  async function detectionLoop() {
    while (showDetection) {
      drawImageScaled2(video, photoCanvas)
      const dataUrl = photoCanvas.toDataURL("image/png");
      const base64_image = dataUrl.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");

      // detect the image
      response = await fetch("/detect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: base64_image }),
      });

      const data = await response.json();
      console.log("Success:", data);
      boundingBoxes = data.boxes || [];

      await new Promise(resolve => setTimeout(resolve, 1));
    }
  }

  // Fill the photo with an indication that none has been
  // captured.

  function clearphoto() {
    const context = photoCanvas.getContext("2d");
    context.fillStyle = "#EEEEEE";
    context.fillRect(0, 0, photoCanvas.width, photoCanvas.height);

    const data = photoCanvas.toDataURL("image/png");
    photo.setAttribute("src", data);
  }

  // Capture a photo by fetching the current contents of the video
  // and drawing it into a canvas, then converting that to a PNG
  // format data URL. By drawing it on an offscreen canvas and then
  // drawing that to the screen, we can change its size and/or apply
  // other changes before drawing it.

  function drawImageScaled2(video, canvas) {
    const context = canvas.getContext("2d");
    console.log(`crop.x: ${crop.x}, crop.y: ${crop.y}, crop.width: ${crop.width}, crop.height: ${crop.height}, 0: ${0}, 0: ${0}, canvas.width: ${canvas.width}, canvas.height: ${canvas.height}, `)
    context.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, canvas.width, canvas.height);

    //
    // const size = Math.min(video.videoWidth, video.videoWidth) / 2;
    // const sx = (video.videoWidth - size) / 2;
    // const sy = (video.videoHeight - size) / 2;
    // const sw = size;
    // const sh = size;
    //
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    // ctx.drawImage(video, sx, sy, sw, sh,
    // 0, 0, canvas.width, canvas.height);
  }

  function classify(base64_image) {
    // classify the image
    fetch("/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: base64_image }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);

        const predictionElement = predictionTemplate.content.cloneNode(true);
        predictionElement.querySelector('.prediction-color-border').style.borderColor = data.color ? data.color.hex : 'gray';
        predictionElement.querySelector('.prediction-source').src = data.source_url;

        data.parts.forEach((part) => {
          console.log("PART ", part.id);
          const partElement = predictionPartTemplate.content.cloneNode(true);
          partElement.querySelector(".prediction-part-image").src = part.url
          partElement.querySelector(".prediction-part-id").innerText = part.id
          partElement.querySelector(".prediction-part-confidence").innerText = Math.round(part.confidence * 100, 0) + "%";
          partElement.querySelector(".prediction-color-swatch").style.backgroundColor = data.color ? data.color.hex : 'transparent';
          partElement.querySelector(".prediction-color-name").innerText = data.color ? data.color.name : '';
          partElement.querySelector(".prediction-color-confidence").innerText = data.color ? Math.round(data.color.confidence * 100, 0) + "%" : '';

          predictionElement.querySelector('.prediction-parts').appendChild(partElement);
        })

        predictionsContainer.innerHTML = '';
        predictionsContainer.appendChild(predictionElement);
      })
      .catch((error) => {
        console.error("Error:", error);
        predictionsContainer.innerText = error;
      });

  }

  function takepicture() {
    console.log("Taking photo...");
    drawImageScaled2(video, photoCanvas)
    const data = photoCanvas.toDataURL("image/png");
    photo.setAttribute("src", data);

    // base64 encode the image
    const base64_image = data.replace(/^data:image\/(png|jpg);base64,/, "");

    classify(base64_image);
  }

  window.onerror = function (msg, url, line, col, error) {
    predictionsContainer.innerText = msg;
  };

  // Set up our event listener to run the startup process
  // once loading is complete.
  window.addEventListener("load", startup, false);

})();
