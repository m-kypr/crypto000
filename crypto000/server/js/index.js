var logElem;
var logInterval;
function updateLog() {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            var text = JSON.parse(xhr.responseText);
            var newText = text.join("<br>");
            document.getElementById('logDiv').innerHTML = newText;
            logElem.innerHtml = newText;
        }
    };
    xhr.open('GET', '/api/log', true);
    xhr.send();
}
function rend() {
    logElem = document.createElement('div');
    logInterval = setInterval(updateLog, 1000);
    logElem.id = "logDiv";
    document.body.appendChild(logElem);
}
document.addEventListener('resize', rend);
rend();
