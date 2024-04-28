# coding=utf-8
import email
import smtplib
import util
import os
import pandas as pd


def get_config():
    mail_config_file_path = os.path.join(util.get_project_dir(), 'config/mail_config.csv')
    df_mail_config = pd.read_csv(mail_config_file_path)
    dict_list = df_mail_config.to_dict(orient='records')
    mail_config = {d['Name']: d['Value'] for d in dict_list}
    return mail_config


def send_mail(subject='SNAIL Notification', content='Hello Snail!'):
    config = get_config()
    sender = config['sender']
    receiver = config['receiver']
    smtp_server = config['smtp_server']
    smtp_port = config['smtp_port']
    password = config['password']

    reply = email.message.EmailMessage()
    reply['From'] = sender
    reply['To'] = receiver
    reply['Subject'] = subject
    reply.set_content(content, subtype='html')
    smtp = smtplib.SMTP_SSL(smtp_server, smtp_port)
    smtp.login(sender, password)
    smtp.send_message(reply)
