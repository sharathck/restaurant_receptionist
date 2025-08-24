// To run this code you need to install the following dependencies:
// npm install @google/genai mime
// npm install -D @types/node
import {
  GoogleGenAI,
  LiveServerMessage,
  MediaResolution,
  Modality,
  Session,
} from '@google/genai';

// DOM Elements
const chatContainer = document.getElementById('chat-container') as HTMLDivElement;
const micButton = document.getElementById('mic-button') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const audioPlayer = document.getElementById('audio-player') as HTMLAudioElement;

// State
let session: Session | undefined = undefined;
const responseQueue: LiveServerMessage[] = [];
let audioParts: string[] = [];
let currentModelTurn: HTMLDivElement | null = null;
let isRecording = false;
let mediaRecorder: MediaRecorder | null = null;
let audioStream: MediaStream | null = null;
let isProcessingTurn = false;


// Utility to add messages to the chat
function addMessage(sender: 'user' | 'model', text: string): HTMLDivElement {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('chat-message', `${sender}-message`);
  messageDiv.textContent = text;
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return messageDiv;
}

// Loading indicator for model response
function showLoadingIndicator(): HTMLDivElement {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('chat-message', 'model-message', 'loading-indicator');
  messageDiv.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div>`;
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return messageDiv;
}

function updateModelMessage(element: HTMLDivElement, newText: string) {
    element.classList.remove('loading-indicator');
    element.innerHTML = '';
    element.textContent = newText;
}

// --- Audio Processing ---

// Converts an ArrayBuffer to a Base64 string.
function arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

interface WavConversionOptions {
  numChannels: number;
  sampleRate: number;
  bitsPerSample: number;
}

function base64ToUint8Array(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

function concatUint8Arrays(arrays: Uint8Array[]): Uint8Array {
    const totalLength = arrays.reduce((acc, val) => acc + val.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const arr of arrays) {
        result.set(arr, offset);
        offset += arr.length;
    }
    return result;
}

function createWavHeader(dataLength: number, options: WavConversionOptions): ArrayBuffer {
  const { numChannels, sampleRate, bitsPerSample } = options;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);

  // RIFF identifier, chunk length, WAVE identifier
  view.setUint8(0, 'R'.charCodeAt(0)); view.setUint8(1, 'I'.charCodeAt(0)); view.setUint8(2, 'F'.charCodeAt(0)); view.setUint8(3, 'F'.charCodeAt(0));
  view.setUint32(4, 36 + dataLength, true);
  view.setUint8(8, 'W'.charCodeAt(0)); view.setUint8(9, 'A'.charCodeAt(0)); view.setUint8(10, 'V'.charCodeAt(0)); view.setUint8(11, 'E'.charCodeAt(0));
  // FMT identifier, chunk length, audio format, channels, sample rate
  view.setUint8(12, 'f'.charCodeAt(0)); view.setUint8(13, 'm'.charCodeAt(0)); view.setUint8(14, 't'.charCodeAt(0)); view.setUint8(15, ' '.charCodeAt(0));
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  // Byte rate, block align, bits per sample
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  // data identifier, chunk length
  view.setUint8(36, 'd'.charCodeAt(0)); view.setUint8(37, 'a'.charCodeAt(0)); view.setUint8(38, 't'.charCodeAt(0)); view.setUint8(39, 'a'.charCodeAt(0));
  view.setUint32(40, dataLength, true);

  return buffer;
}


function parseMimeType(mimeType: string): WavConversionOptions {
  const params = mimeType.split(';').map(s => s.trim());
  const formatPart = params[0].split('/')[1];

  const options: Partial<WavConversionOptions> = {
    numChannels: 1,
    bitsPerSample: 16,
    sampleRate: 24000,
  };

  if (formatPart && formatPart.startsWith('L')) {
    const bits = parseInt(formatPart.slice(1), 10);
    if (!isNaN(bits)) options.bitsPerSample = bits;
  }

  for (const param of params) {
    const [key, value] = param.split('=').map(s => s.trim());
    if (key === 'rate' && value) options.sampleRate = parseInt(value, 10);
  }

  return options as WavConversionOptions;
}

function convertToWav(rawData: string[], mimeType: string): Uint8Array {
  const options = parseMimeType(mimeType);
  const dataChunks = rawData.map(data => base64ToUint8Array(data));
  const data = concatUint8Arrays(dataChunks);
  const wavHeader = new Uint8Array(createWavHeader(data.length, options));
  return concatUint8Arrays([wavHeader, data]);
}

// --- Main App Logic ---

function setUiState(isBusy: boolean, isRecordingActive: boolean = false) {
    micButton.disabled = isBusy && !isRecordingActive;
    if (isRecordingActive) {
        micButton.classList.add('recording');
        statusDiv.textContent = 'Listening... Tap to stop.';
        addMessage('user', '...listening...');
    } else {
        micButton.classList.remove('recording');
        statusDiv.textContent = isBusy ? 'Assistant is responding...' : 'Ready. Tap mic to talk.';
    }
}

async function handleTurn() {
    isProcessingTurn = true;
    setUiState(true);
    let turnComplete = false;
    let turnText = '';
    let audioMimeType = '';

    currentModelTurn = showLoadingIndicator();
  
    while (!turnComplete) {
      const message = await waitMessage();
      
      if (message.serverContent?.modelTurn?.parts) {
          for (const part of message.serverContent.modelTurn.parts) {
              if (part.text) {
                  turnText += part.text;
                  updateModelMessage(currentModelTurn, turnText);
              }
              if (part.inlineData?.data) {
                  audioParts.push(part.inlineData.data);
                  if (!audioMimeType) audioMimeType = part.inlineData.mimeType || '';
              }
          }
      }
      
      if (message.serverContent?.turnComplete) {
        turnComplete = true;
      }
    }
    
    if (audioParts.length > 0 && audioMimeType) {
        const wavData = convertToWav(audioParts, audioMimeType);
        const blob = new Blob([wavData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        audioPlayer.src = url;
        audioPlayer.play();
        audioParts = []; // Reset for next turn
    }
  
    isProcessingTurn = false;
    setUiState(false);
    // Process any queued messages that arrived during this turn
    if (responseQueue.length > 0) processQueueContinuously();
}

function waitMessage(): Promise<LiveServerMessage> {
  return new Promise(resolve => {
    const interval = setInterval(() => {
      const message = responseQueue.shift();
      if (message) {
        clearInterval(interval);
        resolve(message);
      }
    }, 100);
  });
}

function processQueueContinuously() {
    if (isProcessingTurn || responseQueue.length === 0) {
        return;
    }
    handleTurn();
}

// --- Recording Logic ---

async function startRecording() {
    if (!session) return;
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(audioStream);

        mediaRecorder.ondataavailable = async (event) => {
            if (event.data.size > 0 && session) {
                const audioData = await event.data.arrayBuffer();
                const base64Audio = arrayBufferToBase64(audioData);
                session.sendClientContent({
                    audioPart: {
                        data: base64Audio,
                        mimeType: mediaRecorder.mimeType,
                    },
                });
            }
        };

        mediaRecorder.onstart = () => {
            isRecording = true;
            setUiState(false, true);
        };
        
        mediaRecorder.onstop = () => {
            isRecording = false;
            setUiState(true, false); // Set to busy while waiting for final response
        };

        mediaRecorder.start(200); // Send audio chunks every 200ms
    } catch (error) {
        console.error('Microphone access denied:', error);
        statusDiv.textContent = 'Microphone access is required.';
        isRecording = false;
        setUiState(false);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    audioStream = null;
    mediaRecorder = null;
}

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

micButton.addEventListener('click', toggleRecording);

async function main() {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const model = 'gemini-2.5-flash';
    const config = {
      requestModalities: [Modality.AUDIO],
      responseModalities: [Modality.AUDIO, Modality.TEXT],
      mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM,
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: 'Zephyr' },
        },
      },
      systemInstruction: {
        parts: [{
          text: `You are a friendly and professional restaurant receptionist assistant for "Triveni Express". Your primary role is to help customers with inquiries about the menu, prices, reservations, and hours. Always greet the customer with, "Welcome to Triveni Express." Keep your responses concise but helpful, as this is a voice interface.

When customers ask about menu items or prices, refer to the provided restaurant menu data for accurate information. Always greet customers warmly and speak in a conversational tone. If a customer wants to make a reservation, collect their preferred date, time, party size, and contact information. For menu recommendations, ask about their preferences to provide personalized suggestions. If you don't have specific information, politely let them know you'll have a staff member follow up. Confirm important details like reservations by repeating them back. If asked about topics outside your scope (like parking), offer to connect them with a human staff member. Maintain a positive, helpful attitude and thank customers for choosing your restaurant. End conversations by asking if there's anything else you can help them with.

MENU DATA:
\`\`\`json
[
  {
    "restaurant_name": "Triveni Express, Richmod, TX",
    "contact_info": {
      "address": "W. 5th Avenue, Richmond, TX",
      "phone": "(500) 550-5000",
      "website": "www.triveniexpress.com"
    },
    "menu": [
      {
        "category_name": "Appetizers",
        "items": [
          { "name": "Vegetable Samosa", "price": 4.50, "description": "Crispy fried dumplings stuffed with potatoes and vegetables", "tags": ["vegetarian", "appetizer"] },
          { "name": "Lamb Samosa", "price": 5.95, "description": "Crispy fried dumplings stuffed with lamb and vegetables", "tags": ["lamb", "appetizer"] },
          { "name": "Eggplant Pakora", "price": 4.25, "description": "Eggplant and onions coated in a chickpea batter and fried", "tags": ["vegetarian", "appetizer", "eggplant"] },
          { "name": "Chicken Pakora", "price": 5.95, "description": "Chopped chicken and onions coated in a chickpea batter and fried", "tags": ["chicken", "appetizer"] },
          { "name": "Channa Chaat", "price": 5.25, "description": "Chickpeas mixed with potatoes, cucumbers, onions, topped with yogurt, cilantro and a spicy sauce", "tags": ["vegetarian", "appetizer", "chaat", "spicy"] },
          { "name": "Samosa Chaat", "price": 5.95, "description": "Two vegetable samosas, topped with cucumbers, onions, yogurt, cilantro and a spicy sauce", "tags": ["vegetarian", "appetizer", "chaat", "spicy"] }
        ]
      },
      {
        "category_name": "Dosas & Breads",
        "items": [
          { "name": "Butter Naan", "price": 2.00, "description": null, "tags": ["vegetarian", "bread", "naan"] },
          { "name": "Garlic Naan", "price": 3.00, "description": null, "tags": ["vegetarian", "bread", "naan", "garlic"] },
          { "name": "Keema Naan", "price": 3.95, "description": "Flatbread stuffed with spiced minced lamb and cilantro", "tags": ["lamb", "bread", "naan"] },
          { "name": "Cheese Naan", "price": 4.50, "description": "Flatbread baked in a tandoor and stuffed with cheese", "tags": ["vegetarian", "bread", "naan", "cheese"] },
          { "name": "Masala Dosa", "price": 8.45, "description": "Two lentil and rice crepes stuffed with potato, served with coconut chutney and lentil dal", "tags": ["vegetarian", "appetizer", "dosa", "potato", "lentils"] }
        ]
      },
      {
        "category_name": "Tandoori",
        "items": [
          { "name": "Chicken Tandoori", "price": 11.95, "description": "Chicken on-the-bone marinated in yogurt and spices, cooked in a tandoor, and served with sliced red bell peppers and onions", "tags": ["chicken", "tandoori"] },
          { "name": "Tandoori Shrimp", "price": 13.95, "description": "Jumbo shrimp marinated and cooked on skewers in a tandoor oven", "tags": ["seafood", "shrimp", "tandoori"] },
          { "name": "Tandoori Mix Grill", "price": 15.75, "description": "A combination of our most popular tandoori items including shrimp, tandoori chicken tikka, seekh kabob, and boti kabob", "tags": ["tandoori", "mixed_grill", "seafood", "chicken", "lamb", "beef"] }
        ]
      },
      {
        "category_name": "Biryani",
        "items": [
          { "name": "Vegetable Biryani", "price": 9.25, "description": "Vegetable medley topped with cashews and cilantro, served over basmati rice", "tags": ["vegetarian", "biryani", "rice", "contains_nuts"] },
          { "name": "Chicken Biryani", "price": 10.95, "description": "Tender marinated chicken, topped with cashews and cilantro, served over basmati rice", "tags": ["chicken", "biryani", "rice", "contains_nuts"] },
          { "name": "Shrimp and Scallop Biryani", "price": 16.95, "description": "Seared scallops and shrimp, topped with cashews and cilantro, served over basmati rice", "tags": ["seafood", "shrimp", "scallop", "biryani", "rice", "contains_nuts"] },
          { "name": "Paneer Biryani", "price": 9.25, "description": "Homemade cheese and root vegetables, topped with cashews and cilantro, served over basmati rice", "tags": ["vegetarian", "paneer", "cheese", "biryani", "rice", "contains_nuts"] }
        ]
      },
      {
        "category_name": "Lamb Dishes",
        "items": [
          { "name": "Lamb Vindaloo", "price": 14.25, "description": "Boneless chunks of lamb and potatoes cooked in a spicy sauce", "tags": ["lamb", "vindaloo", "spicy", "potato", "main_course"] }
        ]
      },
      {
        "category_name": "Beef Dishes",
        "items": [
          { "name": "Beef Vindaloo", "price": 13.95, "description": "Boneless beef chunks and potato cooked in a spicy sauce", "tags": ["beef", "vindaloo", "spicy", "potato", "main_course"] }
        ]
      },
      {
        "category_name": "Chicken Dishes",
        "items": [
          { "name": "Chicken Vindaloo", "price": 13.95, "description": "Chicken breast and thigh served with potatoes in a spicy sauce", "tags": ["chicken", "vindaloo", "spicy", "potato", "main_course"] },
          { "name": "Butter Chicken", "price": 11.85, "description": "Chicken cooked in a mild buttery curry sauce with fenugreek", "tags": ["chicken", "butter_chicken", "mild", "main_course"] }
        ]
      },
      {
        "category_name": "Vegetable Dishes",
        "items": [
          { "name": "Saag Paneer Curry", "price": 9.95, "description": "Spinach and homemade cheese cooked in a curry sauce", "tags": ["vegetarian", "saag", "paneer", "spinach", "cheese", "curry", "main_course"] }
        ]
      },
      {
        "category_name": "Desserts",
        "items": [
          { "name": "Saffron Kulfi", "price": 4.95, "description": "Traditional saffron ice cream with nuts", "tags": ["vegetarian", "dessert", "ice_cream", "contains_nuts"] },
          { "name": "Gulab Jamun", "price": 5.95, "description": "Milk balls deep-fried, soaked in honey and saffron", "tags": ["vegetarian", "dessert"] }
        ]
      }
    ]
  }
]
\`\`\``,
        }],
      },
    };

    session = await ai.live.connect({
      model,
      callbacks: {
        onopen: function () {
          console.log('Session opened.');
          statusDiv.textContent = 'Connection established. Ready.';
          setUiState(false);
        },
        onmessage: function (message: LiveServerMessage) {
          responseQueue.push(message);
          processQueueContinuously();
        },
        onerror: function (e: ErrorEvent) {
          console.error('Session error:', e.message);
          statusDiv.textContent = `Error: ${e.message}`;
          setUiState(true);
        },
        onclose: function (e: CloseEvent) {
          console.log('Session closed:', e.reason);
          statusDiv.textContent = 'Connection closed. Please refresh.';
          setUiState(true);
        },
      },
      config,
    });
  } catch(error) {
      console.error('Failed to initialize session:', error);
      statusDiv.textContent = 'Initialization failed. Check API Key and console.';
      setUiState(true);
  }
}

main();
