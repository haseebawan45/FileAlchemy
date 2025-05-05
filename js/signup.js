import { db, auth } from "../firebase-config.js";
import { createUserWithEmailAndPassword, deleteUser } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";
import { doc, setDoc, serverTimestamp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-firestore.js";

// Import SweetAlert2
import "https://cdn.jsdelivr.net/npm/sweetalert2@11";

async function saveUserData(event) {
  event.preventDefault();

  // Disable submit button
  const submitButton = document.querySelector('button[type="submit"]');
  submitButton.disabled = true;

  try {
    // Clear previous errors
    document.getElementById('error-message').textContent = '';

    // Get form values
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm_password").value;

    // Validation
    if (!email || !password || !confirmPassword) {
      showError("All fields are required");
      return;
    }

    if (password !== confirmPassword) {
      showError("Passwords do not match");
      return;
    }

    if (password.length < 6) {
      showError("Password must be at least 6 characters");
      return;
    }

    // 1. Create user with Firebase Authentication
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;
    
    try {
      // 2. Save user data in Firestore with server timestamps
      await setDoc(doc(db, "users", user.uid), {
          email: email,
          createdAt: serverTimestamp(),
          lastLogin: serverTimestamp(),
          isAdmin: false
      });

      console.log("User created successfully:", user);

      // ✅ Success Alert
      Swal.fire({
        title: "Account Created!",
        text: "Redirecting to the main page...",
        icon: "success",
        showConfirmButton: false,
        timer: 2000, // Auto close in 2.5 sec
      });

      setTimeout(() => {
        window.location.href = "../pages/main_page.html";
    }, 2000);
    
    } catch (firestoreError) {
      console.error("Firestore Error:", firestoreError);
      showError("Error saving user data. Please try again.");

      // Attempt to delete user from Auth if Firestore fails
      try {
        await deleteUser(user);
        console.log("User deleted due to Firestore error");
      } catch (deleteError) {
        console.error("Failed to delete user:", deleteError);
        showError("Account creation partially failed. Contact support.");
      }
    }

  } catch (error) {
    console.error("Signup Error:", error);
    handleAuthError(error);
  } finally {
    submitButton.disabled = false; // Re-enable button
  }
}

// Function to show error messages using SweetAlert2
function showError(message) {
  Swal.fire({
    title: "Error!",
    text: message,
    icon: "error",
    confirmButtonText: "Try Again",
  });
}

// Function to handle Firebase authentication errors
function handleAuthError(error) {
  let errorMessage = "Error creating account. Please try again.";
  
  switch (error.code) {
    case 'auth/email-already-in-use':
      errorMessage = "Email address is already in use.";
      break;
    case 'auth/invalid-email':
      errorMessage = "Invalid email address.";
      break;
    case 'auth/weak-password':
      errorMessage = "Password is too weak.";
      break;
  }

  showError(errorMessage);
}

// Attach event listener
document.getElementById("signup-form").addEventListener("submit", saveUserData);
