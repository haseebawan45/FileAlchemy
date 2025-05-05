import { auth } from "../firebase-config.js";
import { signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

// SweetAlert2 CDN (Include in your HTML file)
import "https://cdn.jsdelivr.net/npm/sweetalert2@11";

async function loginUser(event) {
  event.preventDefault();

  let email = document.getElementById("email").value;
  let password = document.getElementById("password").value;

  try {
    await signInWithEmailAndPassword(auth, email, password);

    // ✅ Success Alert
    Swal.fire({
      title: "Login Successful!",
      text: "Redirecting...",
      icon: "success",
      showConfirmButton: false,
      timer: 2000, // Auto close in 2 sec
    });

    setTimeout(() => {
      window.location.href = "../pages/main_page.html";
    }, 2000);

  } catch (error) {
    console.error("Error:", error.message);

    // ❌ Error Alert
    Swal.fire({
      title: "Error!",
      text: "Invalid Email or Password!",
      icon: "error",
      confirmButtonText: "Try Again",
    });
  }
}

document.getElementById("login-form").addEventListener("submit", loginUser);
