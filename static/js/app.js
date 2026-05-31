    // ─── Backend base URL ───────────────────────────────────────────
    // When served via Flask (python app.py), Flask serves index.html at /
    // so relative URLs work perfectly. Change only if deploying separately.
    const BASE_URL = '';

    // ─── State ──────────────────────────────────────────────────────
    let currentLanguageCode = 'en';
    let currentFilename = null;      // filename reported by /upload
    let isRecording = false;
    let recognition = null;

    // ─── Theme ──────────────────────────────────────────────────────
    function toggleTheme() {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        updateThemeIcons(isDark);
    }
    function updateThemeIcons(isDark) {
        const cls = isDark ? ['fa-sun','fa-moon'] : ['fa-moon','fa-sun'];
        ['theme-icon','theme-icon-mobile'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove(cls[0]); el.classList.add(cls[1]); }
        });
    }
    (function initTheme() {
        const dark = localStorage.theme === 'dark' || (!localStorage.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
        document.documentElement.classList.toggle('dark', dark);
        updateThemeIcons(dark);
    })();

    // ─── Navigation ─────────────────────────────────────────────────
    function showSection(id) {
        const home = document.getElementById('home-view');
        const dash = document.getElementById('app-dashboard');
        if (id === 'hero' || id === 'home-view') {
            dash.classList.add('hidden'); home.classList.remove('hidden');
        } else {
            home.classList.add('hidden'); dash.classList.remove('hidden');
        }
    }

    // ─── Translations ────────────────────────────────────────────────
    const translations = {
        'en': {
            'navHome':'Home','navDashboard':'Dashboard','navFeatures':'Features','navSignIn':'Sign in',
            'heroBadge':'v2.0 AI Engine Active','heroTitle1':'Understand Legal Documents','heroTitle2':'Instantly with AI',
            'heroSub':'Upload contracts, detect hidden risks, and get simplified explanations using our enterprise-grade AI-powered legal analysis tailored for multiple languages.',
            'btnUpload':'Upload Contract','btnDemo':'Try Demo','uploadTitle':'Upload Legal Document',
            'dragDrop':'Drag & drop your contract here','orClick':'or click to browse',
            'riskTitle':'Contract Risk Analysis','riskScore':'Risk Score','summaryTitle':'Document Summary',
            'chatTitle':'JurisAI Assistant','chatOnline':'Online & Ready',
            'chatPlaceholder':'Ask a question about this contract...',
            'sugg1':'Summarize this document','sugg2':'Explain termination clause','sugg3':'Are there penalties mentioned?',
            'welcome':"Hello! I'm JurisAI. Upload a document or ask me a legal question to get started.",
            'switchMsg':"Language switched to English. How can I assist you today?"
        },
        'hi': {
            'navHome':'होम','navDashboard':'डैशबोर्ड','navFeatures':'विशेषताएं','navSignIn':'साइन इन',
            'heroBadge':'v2.0 AI इंजन सक्रिय','heroTitle1':'कानूनी दस्तावेजों को समझें','heroTitle2':'एआई के साथ तुरंत',
            'heroSub':'अनुबंध अपलोड करें, छिपे हुए जोखिमों का पता लगाएं, और कई भाषाओं के लिए तैयार AI-संचालित कानूनी विश्लेषण का उपयोग करके सरलीकृत स्पष्टीकरण प्राप्त करें।',
            'btnUpload':'अनुबंध अपलोड करें','btnDemo':'डेमो आज़माएं','uploadTitle':'कानूनी दस्तावेज़ अपलोड करें',
            'dragDrop':'अपना अनुबंध यहाँ खींचें और छोड़ें','orClick':'या ब्राउज़ करने के लिए क्लिक करें',
            'riskTitle':'अनुबंध जोखिम विश्लेषण','riskScore':'जोखिम स्कोर','summaryTitle':'दस्तावेज़ सारांश',
            'chatTitle':'JurisAI सहायक','chatOnline':'ऑनलाइन और तैयार',
            'chatPlaceholder':'इस अनुबंध के बारे में एक प्रश्न पूछें...',
            'sugg1':'इस दस्तावेज़ का सारांश दें','sugg2':'समाप्ति खंड की व्याख्या करें','sugg3':'क्या इसमें कोई दंड उल्लिखित है?',
            'welcome':"नमस्ते! मैं JurisAI हूँ। आरंभ करने के लिए एक दस्तावेज़ अपलोड करें या मुझसे कोई कानूनी प्रश्न पूछें।",
            'switchMsg':"भाषा को हिंदी में बदल दिया गया है। आज मैं आपकी कैसे सहायता कर सकता हूँ?"
        },
        'te': {
            'navHome':'హోమ్','navDashboard':'డాష్‌బోర్డ్','navFeatures':'లక్షణాలు','navSignIn':'సైన్ ఇన్',
            'heroBadge':'v2.0 AI ఇంజిన్ సక్రియంగా ఉంది','heroTitle1':'చట్టపరమైన పత్రాలను అర్థం చేసుకోండి','heroTitle2':'AI తో తక్షణమే',
            'heroSub':'బహుళ భాషల కోసం రూపొందించబడిన AI-ఆధారిత న్యాయ విశ్లేషణను ఉపయోగించి కాంట్రాక్ట్‌లను అప్‌లోడ్ చేయండి మరియు సరళీకృత వివరణలను పొందండి.',
            'btnUpload':'కాంట్రాక్ట్ అప్‌లోడ్ చేయండి','btnDemo':'డెమో ప్రయత్నించండి','uploadTitle':'చట్టపరమైన పత్రాన్ని అప్‌లోడ్ చేయండి',
            'dragDrop':'మీ కాంట్రాక్ట్‌ని ఇక్కడ లాగి వదలండి','orClick':'లేదా బ్రౌజ్ చేయడానికి క్లిక్ చేయండి',
            'riskTitle':'కాంట్రాక్ట్ రిస్క్ విశ్లేషణ','riskScore':'రిస్క్ స్కోర్','summaryTitle':'పత్రం సారాంశం',
            'chatTitle':'JurisAI అసిస్టెంట్','chatOnline':'ఆన్‌లైన్ & సిద్ధంగా ఉంది',
            'chatPlaceholder':'ఈ కాంట్రాక్ట్ గురించి ఒక ప్రశ్న అడగండి...',
            'sugg1':'ఈ పత్రాన్ని సంగ్రహించండి','sugg2':'రద్దు నిబంధనను వివరించండి','sugg3':'జరిమానాలు ప్రస్తావించబడ్డాయా?',
            'welcome':"నమస్కారం! నేను JurisAI ని. ప్రారంభించడానికి పత్రాన్ని అప్‌లోడ్ చేయండి లేదా చట్టపరమైన ప్రశ్న అడగండి.",
            'switchMsg':"భాష తెలుగుకి మార్చబడింది. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?"
        },
        'bn': {
            'navHome':'হোম','navDashboard':'ড্যাশবোর্ড','navFeatures':'বৈশিষ্ট্য','navSignIn':'সাইন ইন',
            'heroBadge':'v2.0 এআই ইঞ্জিন সক্রিয়','heroTitle1':'আইনি নথি বুঝুন','heroTitle2':'এআই এর মাধ্যমে সাথে সাথে',
            'heroSub':'চুক্তি আপলোড করুন, লুকানো ঝুঁকি শনাক্ত করুন এবং এআই-চালিত আইনি বিশ্লেষণ ব্যবহার করে সরলীকৃত ব্যাখ্যা পান।',
            'btnUpload':'চুক্তি আপলোড করুন','btnDemo':'ডেমো চেষ্টা করুন','uploadTitle':'আইনি নথি আপলোড করুন',
            'dragDrop':'আপনার চুক্তি এখানে টেনে আনুন এবং ছেড়ে দিন','orClick':'অথবা ব্রাউজ করতে ক্লিক করুন',
            'riskTitle':'চুক্তি ঝুঁকি বিশ্লেষণ','riskScore':'ঝুঁকি স্কোর','summaryTitle':'নথির সারাংশ',
            'chatTitle':'JurisAI সহকারী','chatOnline':'অনলাইন এবং প্রস্তুত',
            'chatPlaceholder':'এই চুক্তি সম্পর্কে একটি প্রশ্ন জিজ্ঞাসা করুন...',
            'sugg1':'এই নথির সারসংক্ষেপ দিন','sugg2':'বাতিলকরণ ধারা ব্যাখ্যা করুন','sugg3':'কোনো জরিমানা উল্লেখ আছে কি?',
            'welcome':"হ্যালো! আমি JurisAI। শুরু করতে একটি নথি আপলোড করুন বা আমাকে একটি আইনি প্রশ্ন জিজ্ঞাসা করুন।",
            'switchMsg':"ভাষা বাংলায় পরিবর্তন করা হয়েছে। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?"
        },
        'mr': {
            'navHome':'मुख्यपृष्ठ','navDashboard':'डॅशबोर्ड','navFeatures':'वैशिष्ट्ये','navSignIn':'साइन इन करा',
            'heroBadge':'v2.0 AI इंजिन सक्रिय','heroTitle1':'कायदेशीर कागदपत्रे समजून घ्या','heroTitle2':'AI सह त्वरित',
            'heroSub':'करार अपलोड करा, लपलेले धोके शोधा आणि अनेक भाषांसाठी तयार AI-चालित कायदेशीर विश्लेषण वापरून सोपे स्पष्टीकरण मिळवा.',
            'btnUpload':'करार अपलोड करा','btnDemo':'डेमो वापरून पहा','uploadTitle':'कायदेशीर दस्तऐवज अपलोड करा',
            'dragDrop':'तुमचा करार येथे ड्रॅग आणि ड्रॉप करा','orClick':'किंवा ब्राउझ करण्यासाठी क्लिक करा',
            'riskTitle':'करार जोखीम विश्लेषण','riskScore':'जोखीम स्कोअर','summaryTitle':'दस्तऐवज सारांश',
            'chatTitle':'JurisAI सहाय्यक','chatOnline':'ऑनलाइन आणि तयार',
            'chatPlaceholder':'या कराराबद्दल एक प्रश्न विचारा...',
            'sugg1':'या दस्तऐवजाचा सारांश द्या','sugg2':'समाप्ती कलमाचे स्पष्टीकरण द्या','sugg3':'काही दंड नमूद केले आहेत का?',
            'welcome':"नमस्कार! मी JurisAI आहे. सुरू करण्यासाठी दस्तऐवज अपलोड करा किंवा मला कायदेशीर प्रश्न विचारा.",
            'switchMsg':"भाषा मराठीत बदलली आहे. आज मी तुम्हाला कशी मदत करू शकतो?"
        },
        'ta': {
            'navHome':'முகப்பு','navDashboard':'டாஷ்போர்டு','navFeatures':'அம்சங்கள்','navSignIn':'உள்நுழைய',
            'heroBadge':'v2.0 AI இயந்திரம் செயலில் உள்ளது','heroTitle1':'சட்ட ஆவணங்களை புரிந்து கொள்ளுங்கள்','heroTitle2':'AI உடன் உடனடியாக',
            'heroSub':'ஒப்பந்தங்களை பதிவேற்றவும், மறைக்கப்பட்ட அபாயங்களைக் கண்டறியவும், AI-இயக்கப்படும் சட்ட பகுப்பாய்வைப் பயன்படுத்தி எளிமையான விளக்கங்களைப் பெறவும்.',
            'btnUpload':'ஒப்பந்தத்தை பதிவேற்றவும்','btnDemo':'டெமோவை முயற்சிக்கவும்','uploadTitle':'சட்ட ஆவணத்தை பதிவேற்றவும்',
            'dragDrop':'உங்கள் ஒப்பந்தத்தை இங்கே இழுத்து விடவும்','orClick':'அல்லது உலாவுவதற்கு கிளிக் செய்யவும்',
            'riskTitle':'ஒப்பந்த அபாய பகுப்பாய்வு','riskScore':'அபாய மதிப்பெண்','summaryTitle':'ஆவண சுருக்கம்',
            'chatTitle':'JurisAI உதவியாளர்','chatOnline':'ஆன்லைனில் & தயார்',
            'chatPlaceholder':'இந்த ஒப்பந்தத்தைப் பற்றி ஒரு கேள்வி கேளுங்கள்...',
            'sugg1':'இந்த ஆவணத்தை சுருக்கவும்','sugg2':'முடித்தல் விதியை விளக்குங்கள்','sugg3':'ஏதேனும் அபராதங்கள் குறிப்பிடப்பட்டுள்ளதா?',
            'welcome':"வணக்கம்! நான் JurisAI. தொடங்க ஒரு ஆவணத்தை பதிவேற்றவும் அல்லது என்னிடம் ஒரு சட்ட கேள்வியை கேட்கவும்.",
            'switchMsg':"மொழி தமிழுக்கு மாற்றப்பட்டுள்ளது. இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?"
        },
        'gu': {
            'navHome':'હોમ','navDashboard':'ડેશબોર્ડ','navFeatures':'વિશેષતાઓ','navSignIn':'સાઇન ઇન કરો',
            'heroBadge':'v2.0 AI એન્જિન સક્રિય','heroTitle1':'કાનૂની દસ્તાવેજો સમજો','heroTitle2':'AI સાથે તરત જ',
            'heroSub':'કરારો અપલોડ કરો, છુપાયેલા જોખમો શોધો અને AI-સંચાલિત કાનૂની વિશ્લેષણ ઉપયોગ કરીને સ્પષ્ટ સ્પષ્ટતા મેળવો.',
            'btnUpload':'કરાર અપલોડ કરો','btnDemo':'ડેમો અજમાવી જુઓ','uploadTitle':'કાનૂની દસ્તાવેજ અપલોડ કરો',
            'dragDrop':'તમારો કરાર અહીં ખેંચો અને છોડો','orClick':'અથવા બ્રાઉઝ કરવા માટે ક્લિક કરો',
            'riskTitle':'કરાર જોખમ વિશ્લેષણ','riskScore':'જોખમ સ્કોર','summaryTitle':'દસ્તાવેજ સારાંશ',
            'chatTitle':'JurisAI સહાયક','chatOnline':'ઓનલાઈન અને તૈયાર',
            'chatPlaceholder':'આ કરાર વિશે પ્રશ્ન પૂછો...',
            'sugg1':'આ દસ્તાવેજનો સારાંશ આપો','sugg2':'સમાપ્તિ કલમ સમજાવો','sugg3':'શું કોઈ દંડનો ઉલ્લેખ છે?',
            'welcome':"નમસ્તે! હું JurisAI છું. પ્રારંભ કરવા માટે દસ્તાવેજ અપલોડ કરો અથવા મને કાનૂની પ્રશ્ન પૂછો.",
            'switchMsg':"ભાષા ગુજરાતીમાં બદલાઈ ગઈ છે. આજે હું તમારી કેવી રીતે મદદ કરી શકું?"
        },
        'kn': {
            'navHome':'ಮುಖಪುಟ','navDashboard':'ಡ್ಯಾಶ್‌ಬೋರ್ಡ್','navFeatures':'ವೈಶಿಷ್ಟ್ಯಗಳು','navSignIn':'ಸೈನ್ ಇನ್',
            'heroBadge':'v2.0 AI ಎಂಜಿನ್ ಸಕ್ರಿಯವಾಗಿದೆ','heroTitle1':'ಕಾನೂನು ದಾಖಲೆಗಳನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ','heroTitle2':'AI ನೊಂದಿಗೆ ತಕ್ಷಣ',
            'heroSub':'ಒಪ್ಪಂದಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ, ಗುಪ್ತ ಅಪಾಯಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಿ ಮತ್ತು AI-ಚಾಲಿತ ಕಾನೂನು ವಿಶ್ಲೇಷಣೆಯನ್ನು ಬಳಸಿಕೊಂಡು ಸರಳೀಕೃತ ವಿವರಣೆಗಳನ್ನು ಪಡೆಯಿರಿ.',
            'btnUpload':'ಒಪ್ಪಂದವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ','btnDemo':'ಡೆಮೊ ಪ್ರಯತ್ನಿಸಿ','uploadTitle':'ಕಾನೂನು ದಾಖಲೆಯನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ',
            'dragDrop':'ನಿಮ್ಮ ಒಪ್ಪಂದವನ್ನು ಇಲ್ಲಿ ಎಳೆಯಿರಿ ಮತ್ತು ಬಿಡಿ','orClick':'ಅಥವಾ ಬ್ರೌಸ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ',
            'riskTitle':'ಒಪ್ಪಂದದ ಅಪಾಯ ವಿಶ್ಲೇಷಣೆ','riskScore':'ಅಪಾಯದ ಸ್ಕೋರ್','summaryTitle':'ದಾಖಲೆ ಸಾರಾಂಶ',
            'chatTitle':'JurisAI ಸಹಾಯಕ','chatOnline':'ಆನ್‌ಲೈನ್ ಮತ್ತು ಸಿದ್ಧವಾಗಿದೆ',
            'chatPlaceholder':'ಈ ಒಪ್ಪಂದದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...',
            'sugg1':'ಈ ದಾಖಲೆಯನ್ನು ಸಂಕ್ಷಿಪ್ತಗೊಳಿಸಿ','sugg2':'ಮುಕ್ತಾಯದ ಷರತ್ತನ್ನು ವಿವರಿಸಿ','sugg3':'ಯಾವುದೇ ದಂಡಗಳನ್ನು ಉಲ್ಲೇಖಿಸಲಾಗಿದೆಯೇ?',
            'welcome':"ನಮಸ್ಕಾರ! ನಾನು JurisAI. ಪ್ರಾರಂಭಿಸಲು ದಾಖಲೆಯನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಅಥವಾ ಕಾನೂನು ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ.",
            'switchMsg':"ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
        },
        'ml': {
            'navHome':'ഹോം','navDashboard':'ഡാഷ്‌ബോർഡ്','navFeatures':'സവിശേഷതകൾ','navSignIn':'സൈൻ ഇൻ ചെയ്യുക',
            'heroBadge':'v2.0 AI എഞ്ചിൻ സജീവമാണ്','heroTitle1':'നിയമപരമായ രേഖകൾ മനസ്സിലാക്കുക','heroTitle2':'AI ഉപയോഗിച്ച് ഉടനടി',
            'heroSub':'കരാറുകൾ അപ്‌ലോഡ് ചെയ്യുക, മറഞ്ഞിരിക്കുന്ന അപകടസാധ്യതകൾ കണ്ടെത്തുക, AI-പവർഡ് നിയമ വിശകലനം ഉപയോഗിച്ച് ലളിതമായ വിശദീകരണങ്ങൾ നേടുക.',
            'btnUpload':'കരാർ അപ്‌ലോഡ് ചെയ്യുക','btnDemo':'ഡെമോ പരീക്ഷിക്കുക','uploadTitle':'നിയമരേഖ അപ്‌ലോഡ് ചെയ്യുക',
            'dragDrop':'നിങ്ങളുടെ കരാർ ഇവിടെ വലിച്ചിടുക','orClick':'അല്ലെങ്കിൽ ബ്രൗസ് ചെയ്യാൻ ക്ലിക്ക് ചെയ്യുക',
            'riskTitle':'കരാർ അപകടസാധ്യത വിശകലനം','riskScore':'റിസ്ക് സ്കോർ','summaryTitle':'രേഖയുടെ സംഗ്രഹം',
            'chatTitle':'JurisAI അസിസ്റ്റന്റ്','chatOnline':'ഓൺലൈനിലും തയ്യാറുമാണ്',
            'chatPlaceholder':'ഈ കരാറിനെക്കുറിച്ച് ഒരു ചോദ്യം ചോദിക്കുക...',
            'sugg1':'ഈ പ്രമാണം സംഗ്രഹിക്കുക','sugg2':'അവസാനിപ്പിക്കൽ വ്യവസ്ഥ വിശദീകരിക്കുക','sugg3':'എന്തെങ്കിലും പിഴകൾ സൂചിപ്പിച്ചിട്ടുണ്ടോ?',
            'welcome':"നമസ്കാരം! ഞാൻ JurisAI ആണ്. ആരംഭിക്കുന്നതിന് ഒരു പ്രമാണം അപ്‌ലോഡ് ചെയ്യുക അല്ലെങ്കിൽ ഒരു നിയമപരമായ ചോദ്യം ചോദിക്കുക.",
            'switchMsg':"ഭാഷ മലയാളത്തിലേക്ക് മാറ്റിയിരിക്കുന്നു. ഇന്ന് നിങ്ങളെ എങ്ങനെ സഹായിക്കാനാകും?"
        }
    };

    // ─── Language ───────────────────────────────────────────────────
    // Elements that map directly to translation keys via data-i18n attribute
    const I18N_MAP = {
        // Nav
        'navHome':      '[data-i18n="navHome"]',
        'navDashboard': '[data-i18n="navDashboard"]',
        'navFeatures':  '[data-i18n="navFeatures"]',
        // Upload panel
        'uploadTitle':  '[data-i18n="uploadTitle"]',
        'dragDrop':     '[data-i18n="dragDrop"]',
        'orClick':      '[data-i18n="orClick"]',
        // Risk panel
        'riskTitle':    '[data-i18n="riskTitle"]',
        'riskScore':    '[data-i18n="riskScore"]',
        // Summary panel
        'summaryTitle': '[data-i18n="summaryTitle"]',
        // Chat
        'chatTitle':    '[data-i18n="chatTitle"]',
        'chatOnline':   '[data-i18n="chatOnline"]',
    };

    function setLanguage(code, name) {
        currentLanguageCode = code;
        document.getElementById('current-lang').innerText = name;

        const t = translations[code] || translations['en'];

        // 1. Update all data-i18n text nodes
        Object.entries(I18N_MAP).forEach(([key, selector]) => {
            document.querySelectorAll(selector).forEach(el => {
                if (t[key]) el.innerText = t[key];
            });
        });

        // 2. Update hero section texts (specific IDs)
        const heroTitle1 = document.getElementById('hero-title1');
        const heroTitle2 = document.getElementById('hero-title2');
        const heroSub    = document.getElementById('hero-sub');
        const heroBadge  = document.getElementById('hero-badge');
        if (heroTitle1) heroTitle1.innerText = t.heroTitle1 || '';
        if (heroTitle2) heroTitle2.innerText = t.heroTitle2 || '';
        if (heroSub)    heroSub.innerText    = t.heroSub    || '';
        if (heroBadge)  heroBadge.innerText  = t.heroBadge  || '';

        // 3. Update button labels
        document.querySelectorAll('[data-i18n="btnUpload"]').forEach(el => { if (t.btnUpload) el.innerText = t.btnUpload; });
        document.querySelectorAll('[data-i18n="btnDemo"]').forEach(el  => { if (t.btnDemo)   el.innerText = t.btnDemo;   });

        // 4. Update chat input placeholder
        const input = document.getElementById('chat-input');
        if (input && t.chatPlaceholder) input.placeholder = t.chatPlaceholder;

        // 5. Update suggestion chips
        const suggIds = ['sugg1','sugg2','sugg3'];
        suggIds.forEach(k => {
            const el = document.querySelector(`[data-i18n="${k}"]`);
            if (el && t[k]) el.innerText = t[k];
        });

        // 6. Update welcome message if it is still the original untouched one
        const welcomeEl = document.getElementById('welcome-msg');
        if (welcomeEl && t.welcome) welcomeEl.innerText = t.welcome;

        // 7. Update chat status line
        const statusEl = document.getElementById('chat-status');
        if (statusEl && t.chatOnline) statusEl.innerText = t.chatOnline;

        // 8. Post a switch notification message in chat
        if (t.switchMsg) addBotMessage(t.switchMsg);
    }
