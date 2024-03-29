import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const widget_ext = {
    name: "gameltb.playground_widgets_IMAGE_ANNOTATE",
    async getCustomWidgets(app) {
        return {
            IMAGE_ANNOTATE(node, inputName, inputData, app) {
                const openEditorDialog = function (node) {
                    const image_annotate_widget = node.widgets ? node.widgets.find((widget) => widget.name === inputName) : null;
                    const image_input = node.inputs ? node.inputs.find((input) => input.name === inputData[1].image_input_name) : null;
                    const image_src_node = node.getInputNode(node.inputs.indexOf(image_input))
                    node.properties.dialogOpened = true;
                    node.dialog = new app.ui.dialog.constructor();

                    // node.dialog.element.style.height = "90%";
                    // node.dialog.element.style.width = "90%";
                    // node.dialog.element.style.display = "block";

                    const closeButton = node.dialog.element.querySelector("button");
                    closeButton.textContent = "CANCEL";

                    closeButton.onclick = () => {
                        node.properties.dialogOpened = false;
                        node.dialog.close();
                    };

                    const container = document.createElement("div");
                    container.id = "drag-image-container";

                    Object.assign(container.style, {
                        display: "flex",
                        gap: "10px",
                        // flexWrap: 'wrap',
                        flexDirection: "row",
                        // justifyContent: 'space-around',
                    });


                    const imageNode = document.createElement("img");
                    if (node.imgs) {
                        imageNode.src = node.imgs[0].src;
                        imageNode.width = node.imgs[0].width;
                        imageNode.height = node.imgs[0].height;
                    }
                    imageNode.id = "canvasImage";

                    const canvasEl = document.createElement("canvas");
                    canvasEl.id = "imageCanvas";

                    Object.assign(canvasEl, {
                        height: `500px`,
                        width: `500px`,
                        style: "border: 1px dotted gray;",
                    });

                    node.properties.canvas = canvasEl;
                    container.append(canvasEl);

                    const showTextEl = document.createElement("textarea");
                    showTextEl.id = "showtext";
                    showTextEl.readOnly = true;

                    const instance = new CanvasSelect(
                        canvasEl,
                    );
                    if (image_src_node) {
                        instance.setImage(image_src_node.imgs[0].src)
                    }
                    const size = [750, 500];
                    canvasEl.width = size[0];
                    canvasEl.height = size[1];

                    canvasEl.style = `width: ${size[0]}px; height: ${size[1]}px;`;
                    canvasEl.style.border = "1px dotted gray";
                    instance.WIDTH = size[0];
                    instance.HEIGHT = size[1];

                    instance.labelMaxLen = 10;
                    instance.setData(JSON.parse(image_annotate_widget.value));
                    // 图片加载完成
                    instance.on("load", (src) => {
                        console.log("image loaded", src);
                    });

                    instance.on("updated", (result) => {
                        console.log('标注结果', result)
                        const list = [...result];
                        list.sort((a, b) => a.index - b.index);
                        showTextEl.value = JSON.stringify(list, null, 2);
                        image_annotate_widget.value = JSON.stringify(list, null, 0);
                    });

                    const modeSelectorEl = document.createElement("select")
                    modeSelectorEl.id = "modeSelector"
                    const options = ['move mode', 'create rect', 'create polygon', 'create dot', 'create line', 'create circle'];
                    for (var i = 0; i < options.length; i++) {
                        var opt = document.createElement('option');
                        opt.value = i;
                        opt.innerHTML = options[i];
                        modeSelectorEl.appendChild(opt);
                    }

                    modeSelectorEl.onchange = (ev) => {
                        instance.createType = parseInt(ev.target.value);
                    };

                    modeSelectorEl.width = 100;

                    const controlContainer = document.createElement("div");
                    Object.assign(controlContainer.style, {
                        display: "flex",
                        flexDirection: "column",
                    });

                    controlContainer.append(modeSelectorEl);

                    container.append(controlContainer);

                    node.dialog.show("");
                    node.dialog.textElement.append(container);

                    Object.assign(showTextEl.style, {
                        flex: 1,
                        margin: "20px",
                    });
                    controlContainer.append(showTextEl);

                    instance.update()
                };

                let input_widget = ComfyWidgets.STRING(node, inputName, ["", { default: "[]", multiline: true }], app);
                let widget = node.addWidget("button", inputName + "_edit_button", "[]", () => {
                    openEditorDialog(node);
                });
                widget.label = "edit image annotate";

                input_widget.widget.linkedWidgets = [widget];
                // hack
                input_widget.widget.type = "STRING"

                return input_widget;
            },
        }
    }
};

app.registerExtension(widget_ext);
