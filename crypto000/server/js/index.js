var logElem;
var logInterval;
function updateLog() {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            var text = JSON.parse(xhr.responseText);
            // console.log(text);
            var newText = text.join("<br>");
            logElem.innerHtml = newText;
        }
    };
    xhr.open('GET', '/api/log', true);
    xhr.send();
}
function rend() {
    logElem = document.createElement('div');
    document.appendChild(logElem);
    logInterval = setInterval(updateLog, 1000);
}
document.addEventListener('resize', rend);
rend();
