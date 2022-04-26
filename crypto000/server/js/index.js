var logElem, tradesElem;
var logInterval;
function update(api) {
    api.forEach(apiName => {
        let xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                var text = JSON.parse(xhr.responseText);
                var arr = [];
                text.forEach(tt => {
                    arr.push(new String(tt));
                });
                document.getElementById(apiName + 'Div').innerHTML = arr.join("<br>");
            }
        };
        xhr.open('GET', '/api/' + apiName, true);
        xhr.send();
    });
}
function rend() {
    let rowElem = document.createElement('div');
    rowElem.className = "row";
    rowElem.style.display = "table";
    rowElem.style.backgroundColor = "red";

    let api = ["log", "trades"];
    api.forEach(apiName => {
        let div = document.createElement('div');
        div.id = apiName + "Div";
        div.className = "row";
        div.style.display = "table-cell";
        div.style.width = "100%";
        div.style.tableLayout = "fixed";
        div.style.borderSpacing = "10px";
        rowElem.appendChild(div);
    });

    logInterval = setInterval(function () {
        update(api);
    }, 2000);

    document.body.appendChild(rowElem);
}
document.addEventListener('resize', rend);
rend();
