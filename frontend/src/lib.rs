use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;
use std::rc::Rc;
use std::cell::RefCell;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct AudioRecorder {
    media_recorder: Option<MediaRecorder>,
    audio_data: Rc<RefCell<Vec<u8>>>,
}

#[wasm_bindgen]
impl AudioRecorder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AudioRecorder {
        AudioRecorder {
            media_recorder: None,
            audio_data: Rc::new(RefCell::new(Vec::new())),
        }
    }

    #[wasm_bindgen]
    pub async fn start_recording(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();
        let media_devices = navigator.media_devices()?;

        let mut constraints = MediaStreamConstraints::new();
        constraints.audio(&JsValue::from(true));
        constraints.video(&JsValue::from(false));

        let promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream = JsFuture::from(promise).await?;
        let media_stream: MediaStream = stream.dyn_into()?;

        // Fix: Use new_with_media_stream instead of new
        let media_recorder = MediaRecorder::new_with_media_stream(&media_stream)?;
        
        let audio_data_ref = self.audio_data.clone();

        let ondataavailable = Closure::wrap(Box::new(move |event: BlobEvent| {
            let blob = event.data();
            let file_reader1 = FileReader::new().unwrap();
            let file_reader2 = FileReader::new().unwrap();
            let file_reader3 = FileReader::new().unwrap();
            let audio_data_clone = audio_data_ref.clone();
            
            let onload = Closure::wrap(Box::new(move |_: Event| {
                if let Ok(array_buffer) = file_reader1.result() {
                    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
                    let mut data = audio_data_clone.borrow_mut();
                    data.extend_from_slice(&uint8_array.to_vec());
                }
            }) as Box<dyn Fn(Event)>);
            
            file_reader2.set_onload(Some(onload.as_ref().unchecked_ref()));
            onload.forget();
            
            let _ = file_reader3.read_as_array_buffer(&blob.unwrap());
        }) as Box<dyn Fn(BlobEvent)>);

        media_recorder.set_ondataavailable(Some(ondataavailable.as_ref().unchecked_ref()));
        ondataavailable.forget();

        media_recorder.start()?;
        self.media_recorder = Some(media_recorder);

        console_log!("Recording started");
        Ok(())
    }

    #[wasm_bindgen]
    pub fn stop_recording(&mut self) -> Result<(), JsValue> {
        if let Some(recorder) = &self.media_recorder {
            recorder.stop()?;
            console_log!("Recording stopped");
        }
        Ok(())
    }

    #[wasm_bindgen]
    pub async fn send_to_tts_api(&self, text: &str, description: &str) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        
        let form_data = FormData::new()?;
        form_data.append_with_str("text", text)?;
        form_data.append_with_str("description", description)?;

        let mut opts = RequestInit::new();
        opts.method("POST");
        opts.body(Some(&form_data));

        let request = Request::new_with_str_and_init("/api/tts", &opts)?;
        
        let response_promise = window.fetch_with_request(&request);
        let response = JsFuture::from(response_promise).await?;
        let response: Response = response.dyn_into()?;

        if response.ok() {
            console_log!("TTS request successful");
            
            // Get audio blob and play it
            let array_buffer_promise = response.array_buffer()?;
            let array_buffer = JsFuture::from(array_buffer_promise).await?;
            
            // Fix: Use document() method instead of document field
            let document = window.document().unwrap();
            let audio: HtmlAudioElement = document.create_element("audio")?.dyn_into()?;
            
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let blob_parts = js_sys::Array::new();
            blob_parts.push(&uint8_array);
            
            let mut blob_options = web_sys::BlobPropertyBag::new();
            blob_options.type_("audio/wav");
            let blob = Blob::new_with_u8_array_sequence_and_options(&blob_parts, &blob_options)?;
            
            let url = Url::create_object_url_with_blob(&blob)?;
            
            audio.set_src(&url);
            let _ = audio.play()?;
            
            console_log!("Audio playing");
        } else {
            console_log!("TTS request failed with status: {}", response.status());
        }

        Ok(())
    }
}
