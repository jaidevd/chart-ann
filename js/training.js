var CLIPBOARD = new CLIPBOARD_CLASS("my_canvas", true);
var CHART_TYPES = []

/**
* image pasting into canvas
*
* @param {string} canvas_id - canvas id
* @param {boolean} autoresize - if canvas will be resized
*/
function CLIPBOARD_CLASS(canvas_id, autoresize) {
  var _self = this;
  var canvas = document.getElementById(canvas_id);
  var ctx = document.getElementById(canvas_id).getContext("2d");

  //handlers
  document.addEventListener('paste', function (e) { _self.paste_auto(e); }, false);

  //on paste
  this.paste_auto = function (e) {
    if (e.clipboardData) {
      var items = e.clipboardData.items;
      if (!items) return;

      //access data directly
      var is_image = false;
      for (var i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          //image
          var blob = items[i].getAsFile();
          var URLObj = window.URL || window.webkitURL;
          var source = URLObj.createObjectURL(blob);
          this.paste_createImage(source);
          is_image = true;
        }
      }
      if(is_image == true){
        e.preventDefault();
      }
    }
  };
  //draw pasted image to canvas
  this.paste_createImage = function (source) {
    var pastedImage = new Image();
    pastedImage.onload = function () {
      if(autoresize == true){
        //resize
        canvas.width = pastedImage.width;
        canvas.height = pastedImage.height;
      }
      else{
        //clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      ctx.drawImage(pastedImage, 0, 0);
    };
    pastedImage.src = source;
  };
}

$(function() {
  fetch('uniq')
  .then(response => response.json())
  .then(function(data) {
    data = _.map(data, 'label')
    $.getJSON('chart_types.json').done(function(e) {
      CHART_TYPES = e
      console.log("...", data, CHART_TYPES)
      $('body').template({CHART_TYPES: [...data, ...CHART_TYPES]})
    })
  })
})

$('body').on('submit', '#imageMetaForm', function(e) {
  e.preventDefault();
  let data = $(this).serializeArray()
  let imageEnc = document.getElementById('my_canvas').toDataURL()
  $.ajax({
    method: "POST", url: 'data',
    success: function() {
      location.reload()
    },
    data: {label: data[0].value, image: imageEnc}
  })
}).on('change', 'select', function() {
  console.log($(this).val())
})
