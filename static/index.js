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
  let startbutton = null;
  let showDetectionCheckbox = null;
  let previewCanvas = null;
  let showDetection = false;
  let boundingBoxes = [];
  let crop = { x: 0, y: 0, width: 300, height: 300 };

  function loadAllClasses() {
    fetch("/classes", { method: "GET", })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);
        document.getElementById("all-classes").innerHTML =
          data.classes
            .map((c) => `<div class="all-classes-item"><img src="/images/${c}.png"/><br/>${c}</div>`).join("");
      })
      .catch((error) => {
        console.error("Error:", error);
        document.getElementById("all-classes").innerText = error;
      });
  }

  function startup() {
    loadAllClasses();
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
    startbutton = document.getElementById("startbutton");

    showDetectionCheckbox = document.getElementById("show-detection-checkbox");

    function updatePreviewCanvas() {
      previewContext.drawImage(video, crop.x, crop.y, crop.width, crop.height, 0, 0, previewCanvas.width, previewCanvas.height);
      boundingBoxes.forEach((box) => {
        previewContext.strokeStyle = 'red';
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

    startbutton.addEventListener(
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
      const base64_image = dataUrl.replace(/^data:image\/(png|jpg);base64,/, "");

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


  function takepicture() {
    drawImageScaled2(video, photoCanvas)
    const data = photoCanvas.toDataURL("image/png");
    photo.setAttribute("src", data);

    // base64 encode the image
    const base64_image = data.replace(/^data:image\/(png|jpg);base64,/, "");

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
        document.getElementById("prediction").innerHTML =
          data.classes
            .filter((c) => c.probability > 1.0)
            .map((c) => `${c.label}: ${c.probability}%<br><img src="/images/${c.label}.png"/>`).join("<br/>");
      })
      .catch((error) => {
        console.error("Error:", error);
        document.getElementById("prediction").innerText = error;
      });
  }

  window.onerror = function (msg, url, line, col, error) {
    document.getElementById("prediction").innerText = msg;
  };

  // Set up our event listener to run the startup process
  // once loading is complete.
  window.addEventListener("load", startup, false);

})();
