const resultsData = [
    {
        preamble: "廷评年少法家流，",
        gt: "廷评年少法家流，心似澄江月正秋。学究天人知远识，权分盐铁许良筹。春风忆酒乌家近，好月论禅谢寺幽。清白比来谁见赏，怜君独有富人侯。",
        baseline: "廷评年少法家流，廷评年少法家流，\n深思明理度朝晖。\n倡言法治彰正义，\n民风淳朴振国威。\n\n论断公平公正，\n德政垂范天下人。\n律己慎行守公道，\n清正廉洁树楷模。\n\n执法如山",
        finetuned: "廷评年少法家流，玉帐横陈金鼎游。玉匣中分千载雪，琼筵上会九秋风。春来见舞朝倾国，秋至闻歌夜不空。为报主恩还下泪，故园无处问芳踪。"
    },
    {
        preamble: "枯木藏龙，",
        gt: "枯木藏龙，雷动必惊。惊者是少，不惊者多。",
        baseline: "枯木藏龙，枯木藏龙，独树一帜；  \n百年风雨，傲视群雄。  \n枝叶苍老，岁月如歌；  \n坚韧不拔，风骨犹存。  \n繁花似锦，生机勃发；  \n枝干参天，山河壮丽。",
        finetuned: "枯木藏龙，孤松隐士。故人归去，犹有高名。古柏寒声，萧疏阴气。秋风不改，长夜难明。"
    },
    {
        preamble: "暗窦养泉容决决，",
        gt: "暗窦养泉容决决，明园护桂放亭亭。历山居处当天半，夏里松风尽足听。",
        baseline: "暗窦养泉容决决，暗窦养泉容决决，流泉石上鸟声低。\n山泉自笑云中语，疑是仙人弄瑶琴。 \n\n月照幽林映翠微，清风徐来舞轻薇。\n潺潺泉水唤客心，石壁流年",
        finetuned: "暗窦养泉容决决，闲园栽竹势盈盈。春来共种垂杨柳，莫学池塘生绿萍。"
    },
    {
        preamble: "碧丛丛，",
        gt: "碧丛丛，高插天，大江翻澜神曳烟。楚魂寻梦风飔然，晓风飞雨生苔钱。瑶姬一去一千年，丁香笻竹啼老猿。古祠近月蟾桂寒，椒花坠红湿云间。",
        baseline: "碧丛丛，翠缕缕，含烟雨。\n山峦叠嶂水潺潺，映照着人间烟火气。\n霞光万道金波涌，璀璨夺目如星辰。\n鸟儿轻啼枝头语，唤醒了沉睡的大地梦。\n\n松柏挺拔迎朝晖",
        finetuned: "碧丛丛，紫阁紫烟中。玉辇回空路渐远，金鞭拂水风生。花满春城游女，柳飞秋浦眠鸥。离宫别馆无消息，唯见天边落照红。"
    },
    {
        preamble: "潘郎美貌谢公诗，",
        gt: "潘郎美貌谢公诗，银印花骢年少时。楚地江皋一为别，晋山沙水独相思。",
        baseline: "潘郎美貌谢公诗，潘郎美貌谢公诗，翩若仙子入画图。\n眉目如画映日月，皓齿银牙动春风。\n青丝长发随风舞，素颜如雪胜桃红。\n举步生姿惊四海，倾城一笑醉千人",
        finetuned: "潘郎美貌谢公诗，莫道无心却有情。独向窗前空独立，夕阳西下鸟声清。"
    },
    {
        preamble: "孤城永巷时相见，",
        gt: "孤城永巷时相见，衰柳闲门日半斜。欲送近臣朝魏阙，犹怜残菊在陶家。",
        baseline: "孤城永巷时相见，孤城深巷夜相逢，  \n月照江水如银波。  \n\n独倚栏杆思故国，  \n寒风瑟瑟入窗愁。  \n\n岁月匆匆如流水，  \n沧桑巨变在眼前。  \n\n多少离别情难舍，  \n唯有明月",
        finetuned: "孤城永巷时相见，一树秋蝉自不闻。莫道此身将老去，十年曾作楚山云。"
    },
    {
        preamble: "别离共成怨，",
        gt: "别离共成怨，衰老更难忘。夜月留同舍，秋风在远乡。朱弦徐向烛，白发强临觞。归献西陵作，谁知此路长。",
        baseline: "别离共成怨，别离共成怨，相思泪满衣。\n\n日月无情换，海枯石烂期。\n\n风华易老去，岁月无常时。\n\n唯有心相连，此情可待时。",
        finetuned: "别离共成怨，一梦是空台。今日花朝月，何时见故乡。无端游子泪，徒尔两心焦。"
    },
    {
        preamble: "纱巾草履竹疏衣，",
        gt: "纱巾草履竹疏衣，晚下香山蹋翠微。一路凉风十八里，卧乘篮轝睡中归。",
        baseline: "纱巾草履竹疏衣，山径松阴水映桥，  \n月光洒落小溪边。  \n野花盛开香气浓，  \n竹林深处鸟语频。  \n\n露珠滴在绿叶间，  \n清风吹过树叶声。  \n微风拂面心舒畅，  \n竹影",
        finetuned: "纱巾草履竹疏衣，云阁烟霞自寂寥。山雨过林红叶合，水云晴树白花飘。寒禽独宿巢中夜，深院无人梦里朝。不向人间能得酒，更将诗句寄归欤。"
    },
    {
        preamble: "苏台踪迹在，",
        gt: "苏台踪迹在，旷望向江滨。往事谁堪问，连空草自春。花疑西子脸，涛想伍胥神。吟尽情难尽，斜阳照路尘。",
        baseline: "苏台踪迹在，曾记云烟远，今朝花落迟。\n\n苍苔阶石砌，古木月光垂。\n\n飞瀑流泉响，幽径鸟语熙。\n\n江山多美事，谁解此心痴。",
        finetuned: "苏台踪迹在，楚泽路遥深。云水俱千里，风帆且一鳞。青林秋草遍，黄叶晚烟侵。唯向西溪宿，相思望月心。"
    },
    {
        preamble: "一别杜陵归未期，",
        gt: "一别杜陵归未期，祗凭魂梦接亲知。近来欲睡兼难睡，夜夜夜深闻子规。",
        baseline: "一别杜陵归未期，二度春风吹剑影，三声雁叫绕山飞。\n\n四海苍茫云雾散，五湖碧波月色明。\n\n六月烟雨江南美，七夕鹊桥相会时。\n\n八方宾客皆欢聚，九霄鸿雁共翱翔",
        finetuned: "一别杜陵归未期，孤舟夜泊水连西。云随日月同飘荡，人逐江湖自可齐。白发若为明镜失，绿窗空隔故山啼。今宵何处相寻处，烟水茫茫秋色迷。"
    }
];

let currentResultIndex = 0;
const totalResults = resultsData.length;

const preambleTextElement = document.getElementById('preamble-text');
const gtTextElement = document.getElementById('gt-text');
const baseTextElement = document.getElementById('base-text');
const finetuneTextElement = document.getElementById('finetune-text');
const slideElement = document.querySelector('.slide');
const comparisonSliderElement = document.querySelector('.comparison-slider'); // For hover pause

const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const currentSlideInfoElement = document.getElementById('current-slide-info');
const autoplayDelayInfoElement = document.getElementById('autoplay-delay-info');

let autoPlayInterval;
const AUTO_PLAY_DELAY = 7000; // 7 seconds per slide
const FADE_DURATION = 600; // Must match CSS transition duration for opacity

autoplayDelayInfoElement.textContent = AUTO_PLAY_DELAY / 1000;

function displayResult(index) {
    const result = resultsData[index];

    slideElement.classList.remove('fade-in');
    slideElement.classList.add('fade-out');

    // Wait for fade-out animation to complete before changing content
    setTimeout(() => {
        preambleTextElement.textContent = result.preamble;
        gtTextElement.textContent = result.gt;
        baseTextElement.textContent = result.baseline;
        finetuneTextElement.textContent = result.finetuned;

        // Ensure the slide is ready for fade-in
        // by removing fade-out and then adding fade-in in the next frame
        requestAnimationFrame(() => {
            slideElement.classList.remove('fade-out');
            slideElement.classList.add('fade-in');
        });
        
        updateNavigation();
    }, FADE_DURATION); 
}

function updateNavigation() {
    currentSlideInfoElement.textContent = `${currentResultIndex + 1} / ${totalResults}`;
    // No need to disable buttons if looping
    // prevBtn.disabled = currentResultIndex === 0;
    // nextBtn.disabled = currentResultIndex === totalResults - 1;
}

function showNextResult() {
    currentResultIndex = (currentResultIndex + 1) % totalResults; // Loop back to start
    displayResult(currentResultIndex);
    resetAutoPlay();
}

function showPrevResult() {
    currentResultIndex = (currentResultIndex - 1 + totalResults) % totalResults; // Loop back to end
    displayResult(currentResultIndex);
    resetAutoPlay();
}

function startAutoPlay() {
    stopAutoPlay(); // Clear any existing interval
    autoPlayInterval = setInterval(showNextResult, AUTO_PLAY_DELAY);
}

function stopAutoPlay() {
    clearInterval(autoPlayInterval);
}

function resetAutoPlay() {
    stopAutoPlay();
    startAutoPlay();
}

// Event Listeners
nextBtn.addEventListener('click', showNextResult);
prevBtn.addEventListener('click', showPrevResult);

// Pause autoplay on hover over the main content area
comparisonSliderElement.addEventListener('mouseenter', stopAutoPlay);
comparisonSliderElement.addEventListener('mouseleave', startAutoPlay);

// Initial display
slideElement.classList.add('fade-in'); // Ensure initial fade-in
displayResult(currentResultIndex);
startAutoPlay(); // Start auto-play on load