import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "gameltb.playground_utils",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "paper_playground_0_ShowString") {
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const r = onExecuted?.apply?.(this, arguments);

        if (this.widgets !== undefined) {
          const pos = this.widgets.findIndex((w) => w.name === "string");
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        }

        for (const list of message.string) {
          const w = ComfyWidgets["STRING"](
            this,
            "string",
            ["STRING", { multiline: true }],
            app
          ).widget;
          w.inputEl.readOnly = true;
          w.inputEl.style.opacity = 0.6;
          w.value = list;
        }

        this.onResize?.(this.size);

        return r;
      };
    }
  },
});
