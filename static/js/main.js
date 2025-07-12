document.addEventListener('DOMContentLoaded', () => {
    // Helper function for messages
    function showMessage(elementId, message, type = 'info') { // type: 'error', 'success', 'loading', 'info'
        const element = document.getElementById(elementId);
        if (!element) return;

        element.textContent = message;
        element.className = `message-box ${type}`; // Add class for styling
        element.style.display = 'block';

        if (type !== 'loading') {
            setTimeout(() => {
                element.style.display = 'none';
                element.textContent = '';
                element.className = 'message-box';
            }, 5000); // Hide after 5 seconds
        }
    }

    // --- Authentication (Local SQLite Backend) ---
    // User session management using localStorage for local host
    function isLoggedIn() {
        return localStorage.getItem('userEmail') !== null;
    }

    function getLoggedInUserEmail() {
        return localStorage.getItem('userEmail');
    }

    function setLoggedInUser(email) {
        localStorage.setItem('userEmail', email);
    }

    function clearLoggedInUser() {
        localStorage.removeItem('userEmail');
    }

    // Redirect logic
    const currentPath = window.location.pathname;
    // Redirect if not logged in and trying to access protected pages
    if (!isLoggedIn() && (currentPath === '/profile' || currentPath === '/detect')) {
        window.location.href = '/login';
    } 
    // Redirect if already logged in and trying to access login/signup/index
    else if (isLoggedIn() && (currentPath === '/' || currentPath === '/login' || currentPath === '/signup')) {
        window.location.href = '/profile';
    }


    // --- Signup Form Handler ---
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const name = document.getElementById('name').value;
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            const institution = document.getElementById('institution').value;
            const class_dept = document.getElementById('class_dept').value;
            const language = document.getElementById('language').value;

            showMessage('message-box', 'Creating account...', 'loading');

            try {
                const response = await fetch('/api/signup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password, name, age, gender, institution, class_dept, language })
                });
                const data = await response.json();

                if (response.ok) {
                    showMessage('message-box', data.message, 'success');
                    setTimeout(() => {
                        window.location.href = '/login';
                    }, 2000);
                } else {
                    showMessage('message-box', data.error || 'Signup failed', 'error');
                }
            } catch (error) {
                showMessage('message-box', `Network error: ${error.message}`, 'error');
                console.error("Signup fetch error:", error);
            }
        });
    }

    // --- Login Form Handler ---
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            showMessage('message-box', 'Logging in...', 'loading');

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await response.json();

                if (response.ok) {
                    setLoggedInUser(email); // Store email in local storage for session
                    showMessage('message-box', data.message, 'success');
                    setTimeout(() => {
                        window.location.href = '/profile';
                    }, 1000);
                } else {
                    showMessage('message-box', data.error || 'Login failed', 'error');
                }
            } catch (error) {
                showMessage('message-box', `Network error: ${error.message}`, 'error');
                console.error("Login fetch error:", error);
            }
        });
    }

    // --- Logout Button Handler ---
    const logoutButtons = document.querySelectorAll('#logout-button, #logout-button-detect');
    logoutButtons.forEach(button => {
        if (button) {
            button.addEventListener('click', async () => {
                try {
                    const response = await fetch('/api/logout', { method: 'POST' });
                    if (response.ok) {
                        clearLoggedInUser();
                        window.location.href = '/login';
                    } else {
                        const data = await response.json();
                        alert("Failed to log out: " + (data.error || "Unknown error"));
                    }
                } catch (error) {
                    console.error("Logout fetch error:", error);
                    alert("Network error during logout: " + error.message);
                }
            });
        }
    });

    // ... (rest of existing code in main.js)

    // --- Profile Display Logic (Modified) ---
    const profileHeading = document.getElementById('profile-heading');
    const profileEmail = document.getElementById('profile-email');
    const profileName = document.getElementById('profile-name');
    const profileAge = document.getElementById('profile-age');
    const profileGender = document.getElementById('profile-gender');
    const profileInstitution = document.getElementById('profile-institution');
    const profileClassDept = document.getElementById('profile-class-dept'); // Corrected ID to match HTML (using hyphen)
    const profileLanguage = document.getElementById('profile-language');
    // New button elements
    const goToDetectorBtn = document.getElementById('go-to-detector-btn');
    const goToEnrollmentBtn = document.getElementById('go-to-enrollment-btn');


    if (profileHeading) { // Check if we are on the profile page
        if (isLoggedIn()) {
            const userEmail = getLoggedInUserEmail();
            // Fetch user data from backend for profile page
            fetch(`/api/profile_data?email=${userEmail}`)
                .then(response => response.json())
                .then(data => {
                    const messageBox = document.getElementById('message-box'); // Unified message box
                    if (data.status === 'success') {
                        const user = data.user;
                        profileHeading.textContent = `Welcome, ${user.name || user.email}!`;
                        profileEmail.textContent = user.email || 'N/A';
                        profileName.textContent = user.name || 'N/A';
                        profileAge.textContent = user.age || 'N/A';
                        profileGender.textContent = user.gender || 'N/A';
                        profileInstitution.textContent = user.institution || 'N/A';
                        profileClassDept.textContent = user.class_dept || 'N/A';
                        profileLanguage.textContent = user.language || 'N/A';

                        // --- NEW LOGIC: Check speaker enrollment status ---
                        fetch(`/api/check_speaker_status?email=${userEmail}`)
                            .then(response => response.json())
                            .then(statusData => {
                                if (statusData.status === 'success') {
                                    if (statusData.is_enrolled) {
                                        goToDetectorBtn.style.display = 'inline-block'; // Show Go to Voice Detector
                                        goToEnrollmentBtn.style.display = 'none'; // Hide Go to Voice Enrollment
                                        showMessage('message-box', 'You are already enrolled. Use the detector!', 'info');
                                    } else {
                                        goToDetectorBtn.style.display = 'none'; // Hide Go to Voice Detector
                                        goToEnrollmentBtn.style.display = 'inline-block'; // Show Go to Voice Enrollment
                                        showMessage('message-box', 'Your voice is not yet enrolled. Please enroll your voice first!', 'info');
                                    }
                                } else {
                                    // Fallback if status check fails
                                    goToDetectorBtn.style.display = 'inline-block';
                                    goToEnrollmentBtn.style.display = 'inline-block'; // Show both as fallback
                                    showMessage('message-box', statusData.error || 'Could not determine enrollment status.', 'error');
                                }
                            })
                            .catch(error => {
                                console.error("Enrollment status fetch error:", error);
                                goToDetectorBtn.style.display = 'inline-block';
                                goToEnrollmentBtn.style.display = 'inline-block'; // Show both as fallback
                                showMessage('message-box', `Network error checking enrollment: ${error.message}`, 'error');
                            });
                        // --- END NEW LOGIC ---

                    } else {
                        showMessage('message-box', data.error || 'Failed to load profile data.', 'error');
                    }
                })
                .catch(error => {
                    showMessage('message-box', `Network error loading profile: ${error.message}`, 'error');
                    console.error("Profile data fetch error:", error);
                });
        } else {
            // Already handled by redirect at the top
        }
    }

// ... (rest of main.js code)


    // --- Audio Recording Variables and Elements (for detect.html) ---
    let mediaRecorder;
    let audioChunks = [];
    let audioBlob; // To store the final recorded audio

    // Timer variables
    let startTime;
    let intervalId;
    const recordingTimer = document.getElementById('recording-timer'); // Get reference to the new span

    // Recording control buttons
    const startRecordBtn = document.getElementById('start-record-btn');
    const stopRecordBtn = document.getElementById('stop-record-btn');
    // NEW BUTTON REFERENCES
    const sendRecordedDetectBtn = document.getElementById('send-recorded-detect-btn');
    const sendRecordedEnrollBtn = document.getElementById('send-recorded-enroll-btn');
    
    const recordStatus = document.getElementById('record-status');
    const audioPlayback = document.getElementById('audio-playback');
    const recordMessageBox = document.getElementById('record-message-box');


    // Helper function to format time as MM:SS
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        const formattedMinutes = String(minutes).padStart(2, '0');
        const formattedSeconds = String(remainingSeconds).padStart(2, '0');
        return `${formattedMinutes}:${formattedSeconds}`;
    }

    // Function to start the timer display
    function startTimer() {
        startTime = Date.now();
        recordingTimer.textContent = '00:00'; // Reset timer display
        intervalId = setInterval(() => {
            const elapsedTime = Math.floor((Date.now() - startTime) / 1000);
            recordingTimer.textContent = formatTime(elapsedTime);
        }, 1000); // Update every second
    }

    // Function to stop and reset the timer display
    function stopTimer() {
        clearInterval(intervalId);
        recordingTimer.textContent = '00:00'; // Reset display to 00:00
    }

    // Helper to disable/enable recorded audio buttons
    function setRecordedAudioButtonsState(enabled) {
        sendRecordedDetectBtn.disabled = !enabled;
        sendRecordedEnrollBtn.disabled = !enabled;
    }


    // --- Start Recording Function ---
    if (startRecordBtn) { // Ensure these elements exist on the page
        startRecordBtn.addEventListener('click', async () => {
            audioChunks = []; // Clear previous recordings
            audioBlob = null;
            audioPlayback.src = '';
            audioPlayback.style.display = 'none';
            setRecordedAudioButtonsState(false); // Disable send buttons initially
            recordMessageBox.style.display = 'none';
            stopTimer(); // Ensure timer is reset if somehow running

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' }); // Use webm for broad compatibility

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    stopTimer(); // Stop the interval when recording stops

                    audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';
                    setRecordedAudioButtonsState(true); // Enable send buttons after recording stops
                    recordStatus.textContent = 'Recording stopped. Play or use the audio.';
                    showMessage('record-message-box', 'Recording finished. Ready to use.', 'success');
                };

                mediaRecorder.start();
                startRecordBtn.disabled = true;
                stopRecordBtn.disabled = false;
                recordStatus.textContent = 'Recording...';
                showMessage('record-message-box', 'Recording started...', 'loading');
                
                startTimer(); // Start the timer when recording actually begins

            } catch (error) {
                console.error('Error accessing microphone:', error);
                showMessage('record-message-box', `Error accessing microphone: ${error.message}. Please allow microphone access.`, 'error');
                startRecordBtn.disabled = false; // Enable if mic access fails
                stopTimer(); // Ensure timer is stopped if mic access fails
            }
        });
    }


    // --- Stop Recording Function ---
    if (stopRecordBtn) {
        stopRecordBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                stopRecordBtn.disabled = true;
                startRecordBtn.disabled = false;
            }
        });
    }

    // --- Send Recorded Audio for Detection ---
    if (sendRecordedDetectBtn) {
        sendRecordedDetectBtn.addEventListener('click', async () => {
            if (!audioBlob) {
                showMessage('record-message-box', 'No audio recorded to send.', 'error');
                return;
            }

            const endpoint = '/api/detect_speaker';
            const formData = new FormData();
            const filename = `recorded_audio_detection_${Date.now()}.webm`;
            
            formData.append('audio_file', audioBlob, filename);
            showMessage('record-message-box', 'Sending recorded audio for detection...', 'loading');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    showMessage('record-message-box', `Predicted Speaker: ${data.predicted_speaker}`, 'success');
                } else {
                    showMessage('record-message-box', data.error || 'Detection failed.', 'error');
                }
            } catch (error) {
                showMessage('record-message-box', `Network error: ${error.message}`, 'error');
                console.error('Recorded audio detection error:', error);
            }
        });
    }

    // --- Send Recorded Audio for Enrollment ---
    if (sendRecordedEnrollBtn) {
        sendRecordedEnrollBtn.addEventListener('click', async () => {
            if (!audioBlob) {
                showMessage('record-message-box', 'No audio recorded to send.', 'error');
                return;
            }

            const endpoint = '/api/enroll_voice';
            const formData = new FormData();
            const filename = `recorded_audio_enrollment_${Date.now()}.webm`;
            
            formData.append('audio_samples', audioBlob, filename); // Append as 'audio_samples' for enrollment
            showMessage('record-message-box', 'Sending recorded audio for enrollment...', 'loading');

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (response.ok) {
                    showMessage('record-message-box', data.message || 'Enrollment successful!', 'success');
                } else {
                    showMessage('record-message-box', data.error || 'Enrollment failed.', 'error');
                }
            } catch (error) {
                showMessage('record-message-box', `Network error: ${error.message}`, 'error');
                console.error('Recorded audio enrollment error:', error);
            }
        });
    }


    // --- Upload Audio File for Detection ---
    const detectUploadForm = document.getElementById('detect-upload-form');
    if (detectUploadForm) {
        detectUploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showMessage('message-box-detect', 'Uploading and detecting voice...', 'loading');

            const file = document.getElementById('detect-audio-file').files[0];
            if (!file) {
                showMessage('message-box-detect', "Please select an audio file to detect.", 'error');
                return;
            }

            const formData = new FormData();
            formData.append('audio_file', file);

            try {
                const response = await fetch('/api/detect_speaker', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (response.ok) {
                    showMessage('message-box-detect', `Predicted Speaker: ${data.predicted_speaker}`, 'success');
                } else {
                    showMessage('message-box-detect', data.error || "An unknown error occurred during detection.", 'error');
                }
            } catch (error) {
                showMessage('message-box-detect', `Network error during detection: ${error.message}`, 'error');
                console.error("Detection fetch error:", error);
            }
        });
    }

    // --- Voice Enrollment Logic (via Upload) ---
    const enrollUploadForm = document.getElementById('enroll-upload-form');
    if (enrollUploadForm) {
        enrollUploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showMessage('message-box-enroll', 'Uploading samples for enrollment...', 'loading');

            const files = document.getElementById('enroll-audio-file').files;
            if (files.length === 0) {
                showMessage('message-box-enroll', "Please select audio files for enrollment.", 'error');
                return;
            }
            if (files.length < 3) {
                showMessage('message-box-enroll', "Please provide at least 3 audio samples for better enrollment.", 'error');
                return;
            }

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('audio_samples', files[i]);
            }

            try {
                const response = await fetch('/api/enroll_voice', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (response.ok) {
                    showMessage('message-box-enroll', data.message || 'Enrollment successful!', 'success');
                } else {
                    showMessage('message-box-enroll', data.error || "An unknown error occurred during enrollment.", 'error');
                }
            } catch (error) {
                showMessage('message-box-enroll', `Network error during enrollment: ${error.message}`, 'error');
                console.error("Enrollment fetch error:", error);
            }
        });
    }
});