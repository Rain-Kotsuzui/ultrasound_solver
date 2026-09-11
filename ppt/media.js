// 图片路径相对于 ppt/index.html；留空时显示空白预留区，不加载假图。
// 填入自己的结果图后刷新页面，并重新运行 verify_presentation.py 导出 PDF。
window.DEFENSE_MEDIA = window.DEFENSE_MEDIA || {
  baseline: "", // 例如 "assets/effects/baseline.png"
  ours: "",     // 例如 "assets/effects/ours.png"
};
