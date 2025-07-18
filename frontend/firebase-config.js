// Import Firebase SDKs
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-firestore.js";

// Firebase Configuration
const firebaseConfig = {
  apiKey: "AIzaSyAryqScHR9B7peDWJCTPUvK3VqF7ytt7f4",
  authDomain: "filealchemy-cbaf9.firebaseapp.com",
  projectId: "filealchemy-cbaf9",
  storageBucket: "filealchemy-cbaf9.appspot.com",
  messagingSenderId: "745651077091",
  appId: "1:745651077091:web:5e41b1be83b1d558ddead6",
  measurementId: "G-6TX9W02JB3"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

// Export instances
export { auth, db };