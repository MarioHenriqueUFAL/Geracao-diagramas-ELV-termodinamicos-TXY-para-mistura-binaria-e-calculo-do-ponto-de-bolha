"""Programa geral para cálculo do ponto de bolha e comparação
de desempenho dos métodos de cálculo de zero de funções (método de newton-raphson,
                                                         método da bissecção,
                                                         método das secantes)"""
#Resolução da questão 1 da disciplina de modelagem
#Autor: Mario Henrique Cosme Juvencio

import numpy as np
import matplotlib.pyplot as plt

class Questao1():
    
    def __init__(self,A1,A2,B1,B2,C1,C2,compA,compB):
        #coeficientes de Antoine
        self.a1 = A1
        self.a2 = A2
        self.b1 = B1
        self.b2 = B2
        self.c1 = C1
        self.c2 = C2
        self.compA = compA
        self.compB = compB
        #atributos para plotar a função objetivo
        self.valoresT = [] #eixo x do gráfico da função objetivo
        self.valoresG = [] #eixo y do gráfico da função objetivo
        #atributos para determinação de desempenho dos métodos
        self.numit_newton = 0
        self.numit_secante = 0
        self.numit_bisseccao = 0
        #atributos para plotar o gráfico TXY
        self.temp_txy = []
        self.x1_txy = []
        self.y1_txy = []
        
    """Funções usadas por todos os métodos"""
    
    def temp_inicial(self,p):
        T1_sat = ((self.b1)/(self.a1 - np.log(p)))- self.c1    
        T2_sat = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        T_0 = np.absolute(T1_sat + T2_sat)/2
        return T_0
        
    def funcao(self,x1,x2,p,T_0): #calculo da função a ser determinada a raiz
        g = x1*(np.exp(self.a1 - ((self.b1)/(T_0+self.c1)))) + x2*(np.exp(self.a2 - ((self.b2)/(T_0+self.c2)))) - p
        g = float(g)
        return g
    
    def funcao_der1(self,x1,x2,p,T_0): #calculo da derivada da função a ser determinada a raiz
        g_linha1 = x1*(np.exp(self.a1 - ((self.b1)/(T_0+self.c1))))*((self.b1)/(T_0+self.c1)**2) + x2*(np.exp(self.a2 - ((self.b2)/(T_0+self.c2))))*((self.b2)/(T_0+self.c2)**2)
        g_linha1 = float(g_linha1)
        return g_linha1
    
    def composicao(self,temp_bolha,x1,p):
        #calcular P1 sat
        P1_sat = np.exp(self.a1 - ((self.b1)/(temp_bolha + self.c1)))
        #calcular composicao
        y1 = (x1*P1_sat)/(p)
        return y1
    
    def error(self,T_1,T_0):
        erro = np.absolute((T_1 - T_0)/(T_1))*100
        erro = float(erro)
        return erro
    
    "bloco de métodos para cálculo de zeros de função"
    
    #método de Newton Raphson
    
    def raiz_newton(self,x1,x2,p,nitmax,erro):
        
        T_0 = self.temp_inicial(p)
        func_0 = self.funcao(x1,x2,p,T_0)
        func_der_0 = self.funcao_der1(x1,x2,p,T_0)
        T_1 = T_0 - ((func_0)/(func_der_0))
        erro_novo = self.error(T_1,T_0)
        T_0 = T_1
        n = 1
        if func_0 == 0:
            return T_0
        else:
            #n = 1
            while erro_novo >= erro and n < nitmax:
                func_0 = self.funcao(x1,x2,p,T_0)
                func_der_0 = self.funcao_der1(x1,x2,p,T_0)
                T_1 = T_0 - ((func_0)/(func_der_0))
                erro_novo = self.error(T_1,T_0)
                T_0 = T_1
                n = n + 1
            if n >= nitmax:
                return print('A função não atendeu os critérios dentro do número máximo de iterações!')
            else:
                self.numit_newton = n #desempenho
                return T_0
    
    #método da secante
    
    def raiz_secante(self,x1,x2,p,nitmax,erro):
        #cálculo das aproximações iniciais, T deve estar entre estes dois limites
        T1_sat = ((self.b1)/(self.a1 - np.log(p)))- self.c1    
        T2_sat = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        Ti1 = T1_sat - ((self.funcao(x1,x2,p,T1_sat)*(T2_sat - T1_sat))/(self.funcao(x1,x2,p,T2_sat)-self.funcao(x1,x2,p,T1_sat)))
        erro_novo = self.error(Ti1,T1_sat)
        Ti_1 = T1_sat
        Ti = Ti1
        func_0 = self.funcao(x1,x2,p,Ti)
        if func_0 == 0:
            return Ti
        else:
            n = 1
            while erro_novo >= erro and n < nitmax:
                Ti1 = Ti - ((self.funcao(x1,x2,p,Ti)*(Ti_1 - Ti))/(self.funcao(x1,x2,p,Ti_1)-self.funcao(x1,x2,p,Ti)))
                erro_novo = self.error(Ti1,Ti)
                Ti_1 = Ti
                Ti = Ti1
                n = n + 1
            
            self.numit_secante = n #desempenho
            return Ti
                
    #método da bissecção
    def raiz_bisseccao(self,x1,x2,p,nitmax,erro):
        #os limites superior e inferior são as temperaturas de saturação
        T1_sat = ((self.b1)/(self.a1 - np.log(p)))- self.c1    
        T2_sat = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        func_1 = self.funcao(x1,x2,p,T1_sat)
        func_2 = self.funcao(x1,x2,p,T2_sat)
        if func_1 == 0:
            return T1_sat
        if func_2 == 0:
            return T2_sat
        elif func_1 * func_2 < 0:
            erro_e = np.absolute((T1_sat - T2_sat)/(T1_sat))*100
            T1 = T1_sat
            T_1 = T2_sat
            n = 0 #desempenho
            while erro_e >= erro:
                n = n + 1 #desempenho
                T_novo = (T1+T_1)/2
                if self.funcao(x1, x2, p, T1) * self.funcao(x1, x2, p, T_novo) < 0 and self.funcao(x1, x2, p, T_1) * self.funcao(x1, x2, p, T_novo) > 0:
                    if self.funcao(x1, x2, p, T1) > 0 and self.funcao(x1, x2, p, T_1) < 0:
                        T_1 = T_novo
                        erro_e = np.absolute((T1 - T_1)/(T1))*100
                    elif self.funcao(x1, x2, p, T1) < 0 and self.funcao(x1, x2, p, T_1) > 0:
                        T_1 = T_novo
                        erro_e = np.absolute((T1 - T_1)/(T1))*100
                elif self.funcao(x1, x2, p, T_1) * self.funcao(x1, x2, p, T_novo) < 0 and self.funcao(x1, x2, p, T1) * self.funcao(x1, x2, p, T_novo) > 0:
                    if self.funcao(x1, x2, p, T_1) > 0 and self.funcao(x1, x2, p, T1) < 0:
                        T1 = T_novo
                        erro_e = np.absolute((T1 - T_1)/(T1))*100
                    elif self.funcao(x1, x2, p, T_1) < 0 and self.funcao(x1, x2, p, T1) > 0:
                        T1 = T_novo
                        erro_e = np.absolute((T1 - T_1)/(T1))*100
            self.numit_bisseccao = n #desempenho
            return (T1 + T_1)/2
    
    
    """Bloco que faz tudo"""
    
    """Para realizar a simulação, basta chamar as funções relatórios fornecendo 
    os parâmetros necessários. Todos os resultados serão calculados e exibidos na tela!"""
    
    # calcular a raiz pelo método de Newton-Raphson
    def solution_newton(self,x1,x2,p,nitmax,erro):
        raiz = self.raiz_newton(x1,x2,p,nitmax,erro)
        return raiz
    
    #calcula a raiz pelo método da secante
    def solution_secante(self,x1,x2,p,nitmax,erro):
        raiz = self.raiz_secante(x1, x2, p, nitmax, erro)
        return raiz
    
    #calcula a raiz pelo método da bissecção
    def solution_bisseccao(self,x1,x2,p,nitmax,erro):
        raiz = self.raiz_bisseccao(x1, x2, p, nitmax, erro)
        return raiz
    
    #calcula a raiz pelo método da bissecção
    #def  solution_bisseccao():
    
    def relatorio_newton(self,x1,x2,p,nitmax,erro):
        temp_bolha = self.solution_newton(x1,x2,p,nitmax,erro)
        aprox_inicial = self.temp_inicial(p)
        y1 = self.composicao(temp_bolha,x1,p)
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        print('Resultados da simulação usando método de Newton Raphson')
        print('Aproximação inicial: '+str(aprox_inicial)+' K')
        print('Para X1 = '+str(x1)+', T de bolha é: '+str(temp_bolha)+' K')
        print('Composição do vapor y1 = '+str(y1)+'.')

    def relatorio_secante(self,x1,x2,p,nitmax,erro):
        temp_bolha = self.solution_secante(x1,x2,p,nitmax,erro)
        T_satcompA = ((self.b1)/(self.a1 - np.log(p)))- self.c1  
        T_satcompB = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        y1 = self.composicao(temp_bolha,x1,p)
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        print('Resultados da simulação usando Método da secante')
        print('Temperatura de saturação do '+self.compA+': '+str(T_satcompA)+' K')
        print('Temperatura de saturação do '+self.compB+': '+str(T_satcompB)+' K')
        print('Para X1 = '+str(x1)+', T de bolha é: '+str(temp_bolha)+' K')
        print('Composição do vapor y1 = '+str(y1)+'.')
        
    def relatorio_bisseccao(self,x1,x2,p,nitmax,erro):
        temp_bolha = self.solution_bisseccao(x1,x2,p,nitmax,erro)
        T_satcompA = ((self.b1)/(self.a1 - np.log(p)))- self.c1  
        T_satcompB = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        y1 = self.composicao(temp_bolha,x1,p)
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        print('Resultados da simulação usando Método da bissecção')
        print('Temperatura de saturação do '+self.compA+': '+str(T_satcompA)+' K')
        print('Temperatura de saturação do '+self.compB+': '+str(T_satcompB)+' K')
        print('Para X1 = '+str(x1)+', T de bolha é: '+str(temp_bolha)+' K')
        print('Composição do vapor y1 = '+str(y1)+'.') 
    def desempenho(self):
        print('------------------------------------------------------')
        print('Desempenho dos métodos iterativos')
        print('Newton-Raphson: '+str(self.numit_newton))
        print('Secante: '+str(self.numit_secante))
        print('Bissecção: '+str(self.numit_bisseccao))
    
    def plotobjetivo(self,p,x1,x2):
        
        """Essa função deve ser utilizada para estudo da viabilidade do uso do método
        da bissecção, observando se o gráfico corta o eixo de temperaturas."""
        
        T_satA = ((self.b1)/(self.a1 - np.log(p)))- self.c1 
        T_satB = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        passo = np.absolute(T_satA - T_satB)/200
        if T_satA < T_satB:
            n = T_satA
            while n <= T_satB:
                self.valoresG.append(self.funcao(x1,x2,p,n))
                self.valoresT.append(n)
                n = n + passo
        elif T_satB < T_satA:
            n = T_satB
            while n <= T_satA:
                self.valoresG.append(self.funcao(x1,x2,p,n))
                self.valoresT.append(n)
                n = n + passo
        #plotagem do gráfico
        fig, graf = plt.subplots()
        graf.plot(self.valoresT,self.valoresG,label='F = G(T)')
        graf.set_xlabel('temperatura (K)')
        graf.set_ylabel('G(T)')
        graf.set_title('Avaliação do comportamento da função objetivo')
        graf.legend()
    
    #função para plotar o gráfico TXY para a mistura binária
    def plotTXY(self,p):
        #definir as temperaturas de saturação pra cálculo dos incrementos
        T_satA = ((self.b1)/(self.a1 - np.log(p)))- self.c1 
        T_satB = ((self.b2)/(self.a2 - np.log(p)))- self.c2
        #definir incremento
        #T2 sempre maior que T1
        if T_satA > T_satB:
            T2 = T_satA
            T1 = T_satB
        elif T_satB > T_satA:
            T2 = T_satB
            T1 = T_satA
        delta_T = (T2 - T1)/200
        while T1 <= T2:
            #cálculos para cada temperatura
            self.temp_txy.append(T1)
            P1_sat = np.exp(self.a1 - ((self.b1)/(T1 + self.c1)))
            P2_sat = np.exp(self.a2 - ((self.b2)/(T1 + self.c2)))
            #cálculo da composição do vapor e do líquido
            x1 = (p - P2_sat)/(P1_sat - P2_sat)
            self.x1_txy.append(x1)
            y1 = (x1*P1_sat)/(p)
            self.y1_txy.append(y1)
            T1 = T1 + delta_T
        #plotagem do gráfico TXY
        fig, graf = plt.subplots()
        graf.plot(self.x1_txy,self.temp_txy,label='Ponto de bolha')
        graf.plot(self.y1_txy,self.temp_txy,label='Ponto de orvalho')
        graf.set_xlabel('x1,y1')
        graf.set_ylabel('T(K)')
        graf.set_title('Gráfico TXY')
        graf.legend()


#composição da fase líquida em termos dos componentes 1 e 2
x1 = 0.5
x2 = 0.5
p = 1.01325 #bar
nitmax = 10000000
erro = 0.0001
#parametros para T em kelvin e P em bar
A1 = 9.2806
A2 = 9.3935
B1 = 2788.51
B2 = 3096.52
C1 = -52.36
C2 = -53.67
#nome dos compostos
compA = 'benzeno'
compB = 'tolueno'

"""Geração e exibição dos resultados"""
#cria o objeto Questao1
obj = Questao1(A1,A2,B1,B2,C1,C2,compA,compB)
obj.relatorio_newton(x1, x2, p, nitmax, erro)
obj.relatorio_secante(x1, x2, p, nitmax, erro)
obj.relatorio_bisseccao(x1, x2, p, nitmax, erro)
obj.plotobjetivo(p,x1,x2)
obj.desempenho()
obj.plotTXY(p)
        
        
        
    
                
            