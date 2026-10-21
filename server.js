const express = require("express");
const bodyParser = require("body-parser");
const session = require("express-session");
const bcrypt = require("bcrypt");
const db = require("./db");

const app = express();
const PORT = 4000;

// ======================================================
// MIDDLEWARE
// ======================================================

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));
app.set("view engine", "ejs");

app.use(
    session({
        secret: "student-secret",
        resave: false,
        saveUninitialized: false
    })
);

// ======================================================
// AUTH MIDDLEWARE
// ======================================================

function isStudent(req, res, next) {
    if (!req.session.student) {
        return res.redirect("/student-login");
    }
    next();
}

// ======================================================
// HOME
// ======================================================

app.get("/", (req, res) => {
    res.redirect("/student-login");
});

// ======================================================
// REGISTER PAGE
// ======================================================

app.get("/student-register", (req, res) => {
    res.render("student-register", { message: "" });
});

// ======================================================
// LOGIN PAGE
// ======================================================

app.get("/student-login", (req, res) => {
    res.render("student-login", { message: "" });
});

// ======================================================
// REGISTER STUDENT
// ======================================================

app.post("/student-register", async (req, res) => {

    const {
        full_name,
        rollno,
        email,
        password,
        confirm_password,
        department,
        semester
    } = req.body;

    try {

        if (password !== confirm_password) {
            return res.render("student-register", {
                message: "Passwords do not match"
            });
        }

        const existingStudent = await db.query(
            `SELECT * FROM students WHERE email = $1`,
            [email]
        );

        if (existingStudent.rows.length > 0) {
            return res.render("student-register", {
                message: "Email already exists"
            });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        await db.query(
            `INSERT INTO students 
            (full_name, rollno, email, password, department, semester)
            VALUES ($1, $2, $3, $4, $5, $6)`,
            [
                full_name,
                rollno,
                email,
                hashedPassword,
                department,
                semester
            ]
        );

        res.render("student-login", {
            message: "Registration Successful"
        });

    } catch (err) {
        console.log(err);
        res.send("Registration Error");
    }

});

// ======================================================
// LOGIN STUDENT
// ======================================================

app.post("/student-login", async (req, res) => {

    const { email, password } = req.body;

    try {

        const result = await db.query(
            `SELECT * FROM students WHERE email = $1`,
            [email]
        );

        if (result.rows.length === 0) {
            return res.render("student-login", {
                message: "Invalid Email"
            });
        }

        const student = result.rows[0];

        const match = await bcrypt.compare(password, student.password);

        if (!match) {
            return res.render("student-login", {
                message: "Wrong Password"
            });
        }

        req.session.student = student;

        res.redirect("/student-dashboard");

    } catch (err) {
        console.log(err);
        res.send("Login Error");
    }

});

// ======================================================
// DASHBOARD (FIXED - YOUR MAIN ISSUE WAS HERE)
// ======================================================

app.get("/student-dashboard", isStudent, async (req, res) => {

    try {

        const studentId = req.session.student.id;

        // FETCH ASSIGNMENTS
        const assignments = await db.query(
            `SELECT * FROM assignments ORDER BY id DESC`
        );

        // FETCH NOTIFICATIONS
        const notifications = await db.query(
            `SELECT * FROM notifications ORDER BY id DESC`
        );

        res.render("student-dashboard", {
            student: req.session.student,
            assignments: assignments.rows || [],
            notifications: notifications.rows || []
        });

    } catch (err) {
        console.log(err);
        res.send("Dashboard Error");
    }

});

// ======================================================
// LOGOUT
// ======================================================

app.get("/student-logout", (req, res) => {
    req.session.destroy(() => {
        res.redirect("/student-login");
    });
});

// ======================================================
// SERVER
// ======================================================

app.listen(PORT, () => {
    console.log(`Student Portal Running on http://localhost:${PORT}`);
});