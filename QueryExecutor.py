import sqlite3

class QueryExecutor:

    def __init__(self, query):
        sql_con = sqlite3.connect("T:\\2018++\\BE\\LP\\LP2\\Data Mining\\dataset\\database.sqlite")
        self.cursor = sql_con.cursor()
        print("Connected to database")
        print(type(query))
        self.cursor.execute(query)

    '''
    def print(self):

        for row in self.cursor:
            print([row[i] for i in range(0, len(row))])
    '''

    def save_to_html(self):

        with open("T:\\2018++\\BE\\LP\\LP2\\trial and error\\templates\\sqloutput.html", "w", encoding="utf-8") as file:
            file.write("<html>")
            file.write("<head>")
            file.write("<title>SQL Query Output</title>")
            file.write("</head>")
            file.write("<body>")
            


        with open("T:\\2018++\\BE\\LP\\LP2\\trial and error\\templates\\sqloutput.html", "a", encoding="utf-8") as file:

            for row in self.cursor:
                file.write("<p>" + ''.join([f"{(row[i])} | " for i in range(0, len(row))]) + "</p>")
                file.write("<p>---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</p>")


        with open("T:\\2018++\\BE\\LP\\LP2\\trial and error\\templates\\sqloutput.html", "a", encoding="utf-8") as file:
            
            file.write("</body")
            file.write("</html>")

'''
query = input("Enter query :")
obj = QueryExecutor(query)
obj.save_to_html()
'''