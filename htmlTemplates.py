css = '''
<style>
.chat-message {
  padding: 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  display: flex;
}

.chat-message.user {
  background-color: #2b313e;
}

.chat-message.bot {
  background-color: #475063;
}

.chat-message .avatar {
  width: 15%;
}

.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}

.chat-message .message {
  width: 85%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="https://i.pinimg.com/1200x/9f/2b/f9/9f2bf9418bf23ddafd13c3698043c05d.jpg">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
  <div class="avatar">
    <img src="https://i.pinimg.com/736x/2b/cc/55/2bcc55b3291035c4e53afbaa5dd1ae83.jpg">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''