#!/usr/bin/env python3
'''
Created on Thu June 5 2018
A Class that defines CDS, Use BootStraping to estimate hazard Rate

__author__ = "Zhejian Peng"
__license__ = MIT
__credits__ = Lehman Brothers, John Hull's 'Option'
__version__ = "1.0.1"
__maintainer__ = "Zhejian Peng"
__email__ = 'zhejianpeng@yahoo.com'
__status__ = "Developing"

__update__ =
'''

# Import library
import numpy as np
import pandas as pd
from scipy.optimize import newton

# yieldcurveTenor, yieldcurveRate, cdsTenors, cdsSpreads,
# premiumFrequency, defaultFrequency,
# accruedPremium, recoveryRate


class cds:
    # global cdsMaturity
    # global spread

    def __init__(self, cdsTenors, cdsSpreads, premiumFrequency, defaultFrequency,
                 accruedPremiumFlag, contract_spread=None, payment_date=None, notional=None):

        self.cdsTenors = cdsTenors
        self.cdsSpreads = cdsSpreads
        self.premiumFrequency = premiumFrequency
        self.defaultFrequency = defaultFrequency
        self.accruedPremiumFlag = accruedPremiumFlag
        self.recoveryRate = recoveryRate
        self.paymentDate = None
        self.spreads = None
        self.notional = notional

        self.contract_spread = contract_spread

    def getDiscountFactor(self, yieldcurveTenor, yieldcurveRate, t):
        '''
        yieldcurveRate: []
        yieldcurveTenor: []
        '''
        result = -1
        min_time_index = 0
        max_time_index = len(yieldcurveTenor) - 1

        if t < 0:
            result = - 1
        elif t == 0:
            result = 1.0
        elif t > 0 and t <= yieldcurveTenor[min_time_index]:
            result = np.exp(-t*yieldcurveRate[0])
        elif t >= yieldcurveTenor[max_time_index]:
            result = np.exp(-t*yieldcurveRate[-1])
        else:
            for i in range(max_time_index):
                # print(t, t >= yieldcurveTenor[i+1] and t < yieldcurveTenor[i+2])
                if t >= yieldcurveTenor[i] and t < yieldcurveTenor[i+1]:
                    # print(yieldcurveTenor[i], yieldcurveTenor[i+1])
                    yield_interpolated = yieldcurveRate[i] + (yieldcurveRate[i+1] - yieldcurveRate[i]) / \
                        (yieldcurveTenor[i+1]-yieldcurveTenor[i]
                         ) * (t-yieldcurveTenor[i])
                    result = np.exp(-t*yield_interpolated)
        return result

    def getSurvivalProbability(self, creditcurveTenor, creditcurveSP, t):
        result = -1
        min_time_index = 0
        max_time_index = len(creditcurveTenor) - 1
        if t < 0:
            result = -1
        elif t == 0:
            result = 1
        elif t > 0 and t <= creditcurveTenor[min_time_index]:
            h = -np.log(creditcurveSP[0] / creditcurveTenor[min_time_index])
            result = np.exp(-h*t)
        elif t == creditcurveTenor[max_time_index]:
            result = creditcurveSP[-1]
        elif t > creditcurveTenor[max_time_index]:
            h = 0
            if len(creditcurveTenor) == 1:
                h = h = - np.log(creditcurveSP[-1]) / \
                    creditcurveTenor[max_time_index]
            else:
                h = - np.log(creditcurveSP[-1]/creditcurveSP[-2]) / \
                    (creditcurveTenor[-1]-creditcurveTenor[-2])
                result = creditcurveSP[-1] * \
                    np.exp(-(t - creditcurveTenor[max_time_index])*h)
        else:  # where t is in between min_time and max_time
            for i in range(max_time_index):
                if t >= creditcurveTenor[i] and t < creditcurveTenor[i+1]:
                    h = -np.log(creditcurveSP[i+1]/creditcurveSP[i]) / \
                        (creditcurveTenor[i+1]-creditcurveTenor[i])
                    result = creditcurveSP[i] * \
                        np.exp(-(t-creditcurveTenor[i])*h)
                    # print('HERE,', creditcurveSP[i])
        return result

    def calculatePremiumLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity,
                            num_premium_year, accruedPremiumFlag, spread, h):
        max_time_index = len(creditcurveTenor) - 1
        if max_time_index > 0 and cdsMaturity <= creditcurveTenor[max_time_index]:
            annuity = 0
            accruedPremium = 0
            N = int(cdsMaturity*num_premium_year)
            for n in range(1, N+1):
                tn = n / num_premium_year
                tnm1 = (n-1) / num_premium_year
                dt = 1.0 / num_premium_year
                annuity += dt * \
                    self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn) * \
                    self.getSurvivalProbability(
                        creditcurveTenor, creditcurveSP, tn)
                if accruedPremiumFlag:
                    accruedPremium += 0.5*dt*self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn)*(
                        self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) - self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
            # print('HERE', spread)
            return spread*(annuity+accruedPremium)
        else:  # When the cds maturity is beyond our current credit curve, we need to estimate the survival probability for payment beyond credit curve
            annuity = 0
            accruedPremium = 0
            N = int(cdsMaturity*num_premium_year)
            M = creditcurveTenor[max_time_index] * \
                num_premium_year if max_time_index > 0 else 0

            for n in range(1, N+1):
                if n <= M:
                    tn = n/num_premium_year
                    tnm1 = (n-1)/num_premium_year
                    dt = 1.0 / num_premium_year

                    annuity += dt * self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn) * \
                        self.getSurvivalProbability(
                            creditcurveTenor, creditcurveSP, tn)
                    if(accruedPremiumFlag):
                        accruedPremium += 0.5*dt*self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn)*(
                            self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) -
                            self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
                else:
                    tn = n/num_premium_year
                    tnm1 = (n-1)/num_premium_year
                    tM = M / num_premium_year
                    dt = 1.0 / num_premium_year

                    survivalProbability_n = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) * \
                        np.exp(-h*(tn - tM))
                    survivalProbability_nm1 = 0
                    if tnm1 <= tM:
                        survivalProbability_nm1 = self.getSurvivalProbability(
                            creditcurveTenor, creditcurveSP, tnm1)
                    else:
                        survivalProbability_nm1 = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) *\
                            np.exp(-h*(tnm1 - tM))

                    annuity += dt * self.getDiscountFactor(
                        yieldcurveTenor, yieldcurveRate, tn)*survivalProbability_n
                    if accruedPremiumFlag:
                        accruedPremium += 0.5*dt * self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn) *\
                            (survivalProbability_nm1-survivalProbability_n)
            # print('spread', spread)
            return spread*(annuity+accruedPremium)

    def calculateDefaultLeg(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate,
                            cdsMaturity, num_default_year, recoveryRate, h):
        # spread = self.spread
        max_time_index = len(creditcurveTenor) - 1
        # accruedPremiumFlag = self.accruedPremium  # True or False
        # print(cdsMaturity, recoveryRate, h, num_default_year)
        if max_time_index > 0 and cdsMaturity <= creditcurveTenor[max_time_index]:
            annuity = 0
            N = int(cdsMaturity * num_default_year)
            for n in range(1, N+1):
                tn = n / num_default_year
                tnm1 = (n-1) / num_default_year
                annuity += self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn)*(
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) -
                    self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
            return (1-recoveryRate)*annuity
        else:  # cdsMaturity > creditcurveTenor[max_time_index]
            annuity = 0
            N = int(cdsMaturity*num_default_year)
            M = creditcurveTenor[max_time_index] * \
                num_default_year if max_time_index > 0 else 0

            for n in range(1, N+1):
                if n <= M:
                    tn = n / num_default_year
                    tnm1 = (n-1) / num_default_year
                    annuity += self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn)*(
                        self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tnm1) -
                        self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tn))
                else:  # n > m
                    tM = M / num_default_year
                    tn = n / num_default_year
                    tnm1 = (n-1) / num_default_year

                    survivalProbability_n = self.getSurvivalProbability(creditcurveTenor, creditcurveSP, tM) *\
                        np.exp(-h*(tn-tM))
                    if tnm1 <= tM:
                        survivalProbability_nm1 = self.getSurvivalProbability(
                            creditcurveTenor, creditcurveSP, tnm1)
                    else:
                        survivalProbability_nm1 = self.getSurvivalProbability(
                            creditcurveTenor, creditcurveSP,  tM) * np.exp(-h*(tnm1 - tM))
                    annuity += self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, tn) * (
                        survivalProbability_nm1 - survivalProbability_n)
                    # print('HERE', annuity)

            return (1-recoveryRate)*annuity

    def bootstrapCDSspread(self, yieldcurveTenor, yieldcurveRate, cdsTenors, cdsSpreads, recoveryRate):
        premiumFrequency = self.premiumFrequency
        defaultFrequency = self.defaultFrequency
        accruedPremiumFlag = self.accruedPremiumFlag

        def objfunFindHazardRate(h, creditcurveSP,  creditcurveTenor, cdsMaturity, spread):
            # print(cdsMaturity)
            premLeg = self.calculatePremiumLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, premiumFrequency,
                                               accruedPremiumFlag, spread, h)
            defaultLeg = self.calculateDefaultLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, cdsMaturity, defaultFrequency,
                                                  recoveryRate, h)
            # print(h, premLeg - defaultLeg)
            return premLeg - defaultLeg

       # yieldcurveLength = len(yieldcurveTenor)
        cdsTenorsLength = len(cdsTenors)

        # newcreditcurveLength = 0
        # newcreditcurve = []
        survprob = [None]*cdsTenorsLength
        hazardRate = [None]*cdsTenorsLength
        # global creditcurveSP
        creditcurveSP = []
        # global creditcurveTenor
        creditcurveTenor = []
        for i in range(cdsTenorsLength):
            # global cdsMaturity
            cdsMaturity = cdsTenors[i]
            # global spread
            spread = cdsSpreads[i]
            # print(cdsMaturity, spread)
            h = newton(objfunFindHazardRate, 0.01881194, args=(
                creditcurveSP, creditcurveTenor, cdsMaturity, spread))
            hazardRate[i] = h
            if i == 0:
                survprob[i] = np.exp(-hazardRate[i]*cdsTenors[i])
            else:
                survprob[i] = survprob[i-1] * \
                    np.exp(-hazardRate[i]*(cdsTenors[i]-cdsTenors[i-1]))
            creditcurveTenor.append(cdsTenors[i])
            creditcurveSP.append(survprob[i])
        return hazardRate, survprob

    def interpolateSpread(self, paymentDate=None):
        if len(paymentDate) != 0:
            self.spreads = paymentDate
        if len(self.spreads) == 0:
            print('Invalid, No Payment Date')
        cdsTenors = self.cdsTenors
        cdsSpreads = self.cdsSpreads

        def helper(cdsTenors, cdsSpreads, t):
            result = -1
            min_time_index = 0
            max_time_index = len(cdsTenors) - 1
            if t < 0:
                result = -1
            elif t == 0:
                result = 0.01
            elif t < cdsTenors[min_time_index]:
                result = 0.01 + (cdsSpreads[1] - cdsSpreads[0]) / \
                    (cdsTenors[1] - cdsTenors[0]) * (t)
            elif t >= cdsTenors[max_time_index]:
                result = cdsSpreads[max_time_index]
            else:  # in between
                for i in range(max_time_index):
                    if t >= cdsTenors[i] and t < cdsTenors[i+1]:
                        result = cdsSpreads[i] + (cdsSpreads[i+1] - cdsSpreads[i]) / (
                            cdsTenors[i+1] - cdsTenors[i]) * (t - cdsTenors[i])

            return result

        inter_spread = []
        for i in days:
            inter_spread.append(helper(cdsTenors, cdsSpreads, i))
        self.interpolatedSpread = inter_spread
        return inter_spread

        # Unit Test Input
    def marktoMarketValue(self, creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate, spread=None, notional=None):
        # for i in
        if self.notional == None:
            notional = notional
        else:
            notional = self.notional
        if self.contract_spread == None:
            spread = spread
        else:
            spread = self.contract_spread
        dl = test_cds.calculateDefaultLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate,
                                          creditcurveTenor[-1], self.defaultFrequency, self.recoveryRate, 1)
        rpv01 = test_cds.calculatePremiumLeg(days, survprob, yieldcurveTenor, yieldcurveRate,
                                             creditcurveTenor[-1], self.premiumFrequency, self.accruedPremiumFlag, 1, 1)

        breakevenSpread = dl / rpv01
        print('The BreakEven Spread is:', breakevenSpread)
        print('RiskyPV01 is :', rpv01)
        return (breakevenSpread - spread)*notional*rpv01

    def __marktoMarketValue(self, yieldcurveTenor, yieldcurveRate, cdsTenors, cdsSpreads, survprob, notional, spread):
        if self.notional == None:
            notional = notional
        else:
            notional = self.notional
        if self.contract_spread == None:
            spread = spread
        else:
            spread = self.contract_spread
        default_prob = [1-x for x in survprob]
        print(default_prob)
        expected_payment = [0.5*x*spread for x in default_prob]
        discount_factor = []
        for i in cdsTenors:
            temp = self.getDiscountFactor(yieldcurveTenor, yieldcurveRate, i)
            discount_factor.append(temp)
        payment_pv = [x*y for x, y in zip(expected_payment, discount_factor)]
        return np.sum(payment_pv)*notional


yieldcurveTenor = [0.5, 1, 2, 3, 4, 5]
yieldcurveRate = [0.01350, 0.01430, 0.0190, 0.02470, 0.02936, 0.03311]
creditcurveTenor = [1, 3, 5, 7]
creditcurveSP = [0.99, 0.98, 0.95, 0.92]
cdsTenors = [1, 2, 3, 4, 5]
cdsSpreads = [0.0110, 0.0120, 0.0130, 0.0140, 0.0150]
premiumFrequency = 4
defaultFrequency = 12
accruedPremiumFlag = True
recoveryRate = 0.40


# Define Unit Test
def unit_test_getDiscountFactor():
    def helper(t, yield_true):
        epsilon = 10**-6
        test_cds = cds(cdsTenors, cdsSpreads, premiumFrequency,
                       defaultFrequency, accruedPremiumFlag, 0.02)
        rate = test_cds.getDiscountFactor(yieldcurveTenor, yieldcurveRate, t)
        print('test t =', t, ', yieldRate = ', rate)
        error = rate - yield_true
        # print(error)
        assert((abs(error) < epsilon))

    test_t = [-1, 0, 1, 2, 3, 4, 5, 6, 1.5]
    yield_ans = [-1, 1.000000, 0.985802, 0.962713,
                 0.928579, 0.889194, 0.847427, 0.819829, 0.975334]
    print('TEST: cds.getDiscountFactor()')
    for i, j in zip(test_t, yield_ans):
        # print(i, j)
        helper(i, j)


def unit_test_getSurvivalProbability():
    def helper(t, sp_true):
        epsilon = 10**-6
        test_cds = cds(cdsTenors, cdsSpreads, premiumFrequency,
                       defaultFrequency, accruedPremiumFlag, 0.02)
        sp = test_cds.getSurvivalProbability(
            creditcurveTenor, creditcurveSP, t)
        print('test t =', t, ', survival probability = ', sp)
        error = sp - sp_true
        # print(error)
        assert((abs(error) < epsilon))

    print('\nTEST: cds.getSurvivalProbability()')

    test_t = [-1, 0, 1, 3, 5, 7, 10, 5.5]
    sp_ans = [-1, 1, 0.990000, 0.980000,
              0.950000, 0.920000, 0.876767, 0.942410]
    for i, j in zip(test_t, sp_ans):
        helper(i, j)


def unit_test_calculatePremiumLeg():
    def helper(param, pl_ture):
        epsilon = 10**-6
        maturity, spread, accruedPremium, h = param
        test_cds = cds(cdsTenors, cdsSpreads, premiumFrequency,
                       defaultFrequency, accruedPremiumFlag, 0.02)
        pl = test_cds.calculatePremiumLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate,
                                          maturity, 4, accruedPremium, spread, h)
        print('test param =', param, ', premium leg = ', pl)
        error = pl - pl_ture
        assert(abs(error) < epsilon)

    param = [[4, 0.02, True, 0.01881194], [-1, 0.02,
                                           True, 0.01], [4, 0.02, False, 0.01], [4, 0.1, True, 0.1], [8, 0.1, True, 0.1]]
    pl_ans = [0.0750303, 0, 0.074947, 0.375151623674, 0.678580]

    print('\nTEST: cds.calculatePremiumLeg()')
    for i, j in zip(param, pl_ans):
        helper(i, j)


def unit_test_calculateDefaultLeg():

    def helper(param, dl_true):
        epsilon = 10**-6
        maturity, defaultFrequency, recoveryRate, h = param
        test_cds = cds(cdsTenors, cdsSpreads, premiumFrequency,
                       defaultFrequency, accruedPremiumFlag, 0.02)
        dl = test_cds.calculateDefaultLeg(creditcurveTenor, creditcurveSP, yieldcurveTenor, yieldcurveRate,
                                          maturity, defaultFrequency, recoveryRate, h)
        print('test param = ', param, ', default leg =', dl)
        error = dl - dl_true
        assert(abs(error) < epsilon)

    param = [[4, 12, 0.4, 0.01881194], [0, 0, 0, 0],
             [-1, 12, 0.4, 0.01], [8, 12, 0.4, 0.01], [10, 12, 0.5, 0.01], [3.4, 100, 0.6, 0.1]]
    pl_ans = [0.019947, 0, 0, 0.046710, 0.045612, 0.010053]

    print('\nTEST: cds.calculateDefaultLeg()')
    for i, j in zip(param, pl_ans):
        helper(i, j)


# Start Unit Test
unit_test_getDiscountFactor()
unit_test_getSurvivalProbability()
unit_test_calculatePremiumLeg()
unit_test_calculateDefaultLeg()

# test BootStraping
print('\nTEST: cds.bootstrapCDSspread()')

print()
test_cds = cds(cdsTenors, cdsSpreads, premiumFrequency,
               defaultFrequency, True, 0.02)
rate, result = test_cds.bootstrapCDSspread(
    yieldcurveTenor, yieldcurveRate, cdsTenors, cdsSpreads, recoveryRate)
print(rate, result)


day_count = [0.261111, 0.252778, 0.252778, 0.252778, 0.252778, 0.252778, 0.252778, 0.252778, 0.255556, 0.252778,
             0.250000, 0.255556, 0.255556, 0.252778, 0.250000, 0.255556, 0.255556]
days = np.cumsum(day_count)
# print(days)
interp_spread = test_cds.interpolateSpread(days)
# assert(interp_spread == [0.010261111, 0.010513889, 0.010766667000000001, 0.011019444999999999, 0.011272223,
#                          0.011525001, 0.011777779, 0.012030557000000001, 0.012286113, 0.012538891, 0.012788891,
#                          0.013044446999999999, 0.013300002999999999, 0.014313892999999999])

# print((interp_spread))
# Calculate The Survival Probability of Lehman's Contract:
print('\nCalculate the Survival Probability in Lehman Contrat')

hazard_rate, survprob = test_cds.bootstrapCDSspread(
    yieldcurveTenor, yieldcurveRate, days, interp_spread, recoveryRate)
print('Hazard_rate', survprob)

print('\nCalculate the MTM Value in Lehman Contrat')

temp = test_cds.marktoMarketValue(
    days, survprob, yieldcurveTenor, yieldcurveRate, 0.02, 100000000)
print('The Current Value of Contract is', temp)


# cdsValue = test_cds.marktoMarketValue(
#     yieldcurveTenor, yieldcurveRate, days, interp_spread, survprob, 1, 0.02)
# print(cdsValue)
import matplotlib.pyplot as plt
# print(days)
# plt.plot(days, survprob)
# plt.show()
