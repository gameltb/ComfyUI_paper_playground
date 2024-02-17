import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";

app.registerExtension({
  name: "gameltb.diffusers",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (
      nodeData.name === "DiffusersPipelineComponentShow" ||
      nodeData.name === "DiffusersPipelineListAdapters"
    ) {
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const r = onExecuted?.apply?.(this, arguments);

        if (this.widgets !== undefined) {
          const pos = this.widgets.findIndex(
            (w) => w.name === "components_map"
          );
          if (pos !== -1) {
            for (let i = pos; i < this.widgets.length; i++) {
              this.widgets[i].onRemove?.();
            }
            this.widgets.length = pos;
          }
        }

        for (const list of message.components_map) {
          const w = ComfyWidgets["STRING"](
            this,
            "components_map",
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
