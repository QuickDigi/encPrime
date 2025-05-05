const crypto = require('crypto');

class Encrypt {
    morse = (text, reverse = false) => {
        text = String(text);
        const pattle = {
            A: ".-",
            B: "-...",
            C: "-.-.",
            D: "-..",
            E: ".",
            F: "..-.",
            G: "--.",
            H: "....",
            I: "..",
            J: ".---",
            K: "-.-",
            L: ".-..",
            M: "--",
            N: "-.",
            O: "---",
            P: ".--.",
            Q: "--.-",
            R: ".-.",
            S: "...",
            T: "-",
            U: "..-",
            V: "...-",
            W: ".--",
            X: "-..-",
            Y: "-.--",
            Z: "--..",
            "1": ".----",
            "2": "..---",
            "3": "...--",
            "4": "....-",
            "5": ".....",
            "6": "-....",
            "7": "--...",
            "8": "---..",
            "9": "----.",
            "0": "-----",
            ".": ".-.-.-",
            ",": "--..--",
            "?": "..--..",
            "'": ".----.",
            "!": "-.-.--",
            "/": "-..-.",
            "(": "-.--.",
            ")": "-.--.-",
            "&": ".-...",
            ":": "---...",
            ";": "-.-.-.",
            "=": "-...-",
            "+": ".-.-.",
            "-": "-....-",
            _: "..--.-",
            $: "...-..-",
            "@": ".--.-.",
        }

        if (reverse) {
            let result = ""
            let morseArray = text.trim().split(" ") // تقسيم مورس إلى أجزاء مفصولة بمسافات
            morseArray.forEach(code => {
                let char = Object.keys(pattle).find(key => pattle[key] === code) // البحث عن الحرف في القاموس
                if (char) {
                    result += char
                } else {
                    result += " " // إذا مفيش تطابق للشفرة، إضافة فراغ
                }
            })
            return result
        } else {
            // الكود الخاص بالتحويل من نص إلى مورس (كما كان قبل ذلك)
            let result = ""
            text = text.toUpperCase().split("") // تحويل النص إلى أحرف كبيرة
            text.forEach(char => {
                if (pattle[char]) {
                    result += pattle[char] + " "
                } else {
                    result += " " // وضع فراغ للأحرف التي لا توجد لها شفرة مورس
                }
            })
            return result.trim()
        }
    }

    /**
   * Executes a function in parallel across multiple workers.
   *
   * @param {String} text - The text to encrypt or decrypt.
   * @param {Object} options - The encryption/decryption options.
   * @param {string} key - The encryption/decryption key.
   * @param {boolean} [reverse=false] - Set to true to decrypt the text.
   * @returns {Promise<string>} - The encrypted or decrypted text.
   *
   * @example
   *  const key = "your-encryption-key"; // Replace with your own encryption key
   *    const text = "Hello, World!";
   *
   *    // Encrypt the text
   *    const encryptedText = encryptor.aes(text, {
   *        key,
   *    });
   *    console.log("Encrypted:", encryptedText);
   *
   *    // Decrypt the text
   *    const decryptedText = encryptor.aes(encryptedText, {
   *        key,
   *        reverse: true,  // Set to true to decrypt the text
   *    });
   *    console.log("Decrypted:", decryptedText);
   */
    aes = (text, { key, reverse }) => {
        const keyBuffer = Buffer.from(key).slice(0, 32); // Ensure the key is 256 bits (32 bytes)
        if (!reverse) {
            const iv = crypto.randomBytes(16); // Generate a random IV
            const cipher = crypto.createCipheriv('aes-256-cbc', keyBuffer, iv);
            let encrypted = cipher.update(text, 'utf8', 'hex');
            encrypted += cipher.final('hex');
            return iv.toString('hex') + ':' + encrypted; // Prepend IV to the encrypted text
        } else if (reverse) {
            const textParts = text.split(':');
            const iv = Buffer.from(textParts.shift(), 'hex'); // Extract IV
            const encryptedText = textParts.join(':');
            const decipher = crypto.createDecipheriv('aes-256-cbc', keyBuffer, iv);
            let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
            decrypted += decipher.final('utf8');
            return decrypted;
        }
    }
}

module.exports = Encrypt 