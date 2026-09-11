"""Check the offline presentation and export its PDF."""

import json
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "verification"


def main():
    OUTPUT.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
        errors = []
        requests = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto((HERE / "index.html").as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(2400)
        slide_count = page.locator(".slide").count()
        assert slide_count == 8
        assert page.locator("#algorithm-chart .metric-row").count() == 3
        assert page.locator("#ablation-chart .metric-row").count() == 5
        assert page.locator("#array-drawing span").count() == 256
        configured_media = page.evaluate("Object.values(window.DEFENSE_MEDIA).filter(Boolean).length")
        page.wait_for_function(
            "expected => document.querySelectorAll('.effect-slot.loaded').length === expected",
            arg=configured_media,
        )
        assert page.locator(".effect-slot:not(.loaded)").count() == 2 - configured_media
        page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
        assert page.locator(".loss-figure img").count() == 2
        checks = []
        for width, height in [(1600, 900), (1920, 1080), (1366, 768), (390, 844), (844, 390)]:
            page.set_viewport_size({"width": width, "height": height})
            for number in range(1, slide_count + 1):
                page.keyboard.press(str(number))
                page.wait_for_timeout(70)
                result = page.evaluate("""() => {
                    const slide = document.querySelector('.slide.active');
                    const sr = slide.getBoundingClientRect();
                    const footer = slide.querySelector('footer').getBoundingClientRect();
                    const scale = sr.width / 1600;
                    const issues = [];
                    for (const node of slide.querySelectorAll('h1,h2,h3,p,article,figure,img,.metric-row,math')) {
                        const r = node.getBoundingClientRect();
                        if (node.closest('template')) continue;
                        if (r.left < sr.left - 1 || r.right > sr.right + 1 || r.top < sr.top - 1 ||
                            r.bottom > sr.bottom + 1) issues.push('outside slide: ' + node.textContent.slice(0,70));
                        if (!node.closest('footer') && r.bottom > footer.top + 2 &&
                            r.top < footer.bottom) issues.push('footer collision: ' + node.textContent.slice(0,70));
                        if (node.clientWidth && node.scrollWidth > node.clientWidth + 3)
                            issues.push('horizontal overflow: ' + node.textContent.slice(0,70));
                    }
                    const blocks = [...slide.querySelectorAll('h1,h2,h3,p,math,.array-stats,.bottom-thesis,.gradient-bottom,.closing-statement,.experiment-footnotes')];
                    for (let i=0;i<blocks.length;i++) for(let j=i+1;j<blocks.length;j++) {
                        const a=blocks[i],b=blocks[j];
                        if(a.contains(b)||b.contains(a))continue;
                        const x=a.getBoundingClientRect(),y=b.getBoundingClientRect();
                        if(Math.min(x.right,y.right)-Math.max(x.left,y.left)>4*scale &&
                           Math.min(x.bottom,y.bottom)-Math.max(x.top,y.top)>4*scale)
                           issues.push('content collision: '+a.textContent.trim().slice(0,35)+' / '+b.textContent.trim().slice(0,35));
                    }
                    const chartTicks = [...slide.querySelectorAll('.metric-row')].map(row => ({
                        text:row.textContent, overlap:[...row.children].some((n,i,a) =>
                          i>0 && n.getBoundingClientRect().left < a[i-1].getBoundingClientRect().right-1)
                    }));
                    return {issues,chartTicks, ratio:sr.width/sr.height, scale,
                      pageOverflow:document.documentElement.scrollWidth>innerWidth ||
                                   document.documentElement.scrollHeight>innerHeight};
                }""")
                checks.append({"viewport": [width, height], "slide": number, **result})
                if width == 1600:
                    page.screenshot(path=str(OUTPUT / f"slide-{number:02d}.png"))
            assert page.locator(".slide.active").get_attribute("data-title") == "硬件部署与结论"
        page.set_viewport_size({"width": 1600, "height": 900})
        page.keyboard.press("Home")
        page.keyboard.press("ArrowLeft")
        assert page.locator("#previous").is_disabled()
        page.keyboard.press("ArrowRight")
        assert page.locator(".slide.active").get_attribute("data-title") == "三维物理求解器"
        page.keyboard.press("n")
        assert page.locator("#notes-dialog").is_visible()
        page.keyboard.press("Escape")
        assert not page.locator("#notes-dialog").is_visible()
        page.keyboard.press("o")
        page.locator("#overview-list button").nth(6).click()
        assert page.locator(".slide.active").get_attribute("data-title") == "共同质量数值对比"
        page.keyboard.press("End")
        assert page.locator("#next").is_disabled()
        page.keyboard.press("Home")
        page.keyboard.press("f")
        page.wait_for_function("!!document.fullscreenElement")
        page.keyboard.press("f")
        page.wait_for_function("!document.fullscreenElement")
        page.goto((HERE / "defense_5min.html").as_uri() + "#3")
        page.wait_for_url("**/index.html#3")
        assert page.locator(".slide.active").get_attribute("data-title") == "响应基与伴随梯度"
        page.goto((HERE / "index.html").as_uri() + "#invalid")
        assert page.locator(".slide.active").get_attribute("data-title") == "问题与主张"
        page.wait_for_timeout(2400)
        # Exercise user-supplied media separately; never export the test images.
        media_page = browser.new_page(viewport={"width": 1600, "height": 900})
        media_page.add_init_script("""window.DEFENSE_MEDIA = {
          baseline: "assets/algorithm_loss_dashboard.png",
          ours: "assets/ablation_loss_dashboard.png"
        };""")
        media_page.goto((HERE / "index.html").as_uri() + "#4")
        media_page.wait_for_function("document.querySelectorAll('.effect-slot.loaded').length === 2")
        assert media_page.locator(".effect-slot img").count() == 2
        assert media_page.locator(".effect-slot .slot-label").count() == 0
        media_page.close()
        pixels = page.evaluate("""() => [...document.querySelectorAll('canvas')].map(c => {
          const a = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
          let nonblank=0;
          for(let i=3;i<a.length;i+=4) if(a[i]>0)nonblank++;
          return {id:c.id,nonblank};
        })""")
        assert all(item["nonblank"] > 0 for item in pixels)
        assert not errors, errors
        assert not [url for url in requests if url.startswith(("http:", "https:"))], requests
        page.pdf(path=str(HERE / "defense_5min.pdf"), prefer_css_page_size=True, print_background=True)
        page.emulate_media(media="print")
        assert all(page.locator(".slide").nth(i).is_visible() for i in range(slide_count))
        page.emulate_media(media="screen")
        browser.close()
    report = {"checks": checks, "console_errors": errors, "canvas_pixels": pixels,
              "external_requests": [], "interaction_checks": "passed"}
    (OUTPUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    contact = Image.new("RGB", (1280, ((slide_count + 1) // 2) * 384), "#dadce1")
    draw = ImageDraw.Draw(contact)
    for i in range(slide_count):
        with Image.open(OUTPUT / f"slide-{i+1:02d}.png") as image:
            thumb = ImageOps.contain(image, (624, 351))
            x, y = (i % 2) * 640 + 8, (i // 2) * 384 + 24
            contact.paste(thumb, (x, y))
            draw.text((x, y - 18), f"{i+1:02d}", fill="#333333")
    contact.save(OUTPUT / "contact-sheet.png")
    issues = [row for row in checks if row["issues"] or row["pageOverflow"]]
    print(json.dumps({"issues": issues, "checks": len(checks), "pdf": "ppt/defense_5min.pdf"}, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
